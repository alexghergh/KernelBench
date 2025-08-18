import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,             # *fp32, input tensor
    out_ptr,           # *fp32, output tensor
    N,                 # int32, batch size
    C,                 # int32, number of channels
    L_IN,              # int32, input sequence length
    STRIDE,            # int32, stride
    PADDING,           # int32, padding
    L_OUT,             # int32, output sequence length
    BLOCK_OUT: tl.constexpr,    # number of output elements each program processes
    KERNEL_SIZE: tl.constexpr,  # pooling window
    DILATION: tl.constexpr,     # dilation
):
    pid_nc = tl.program_id(0)        # parallel over (N * C)
    pid_block = tl.program_id(1)     # parallel over blocks of output positions

    row_in_offset = pid_nc * L_IN    # offset to the beginning of (n, c) row in input
    row_out_offset = pid_nc * L_OUT  # offset to the beginning of (n, c) row in output

    block_start = pid_block * BLOCK_OUT
    out_pos = block_start + tl.arange(0, BLOCK_OUT)
    out_mask = out_pos < L_OUT

    # Initialize maxima with -inf
    max_val = tl.full((BLOCK_OUT,), -float("inf"), tl.float32)

    for k in range(KERNEL_SIZE):
        in_idx = out_pos * STRIDE - PADDING + k * DILATION
        in_mask = (in_idx >= 0) & (in_idx < L_IN) & out_mask
        vals = tl.load(x_ptr + row_in_offset + in_idx, mask=in_mask, other=-float("inf"))
        max_val = tl.maximum(max_val, vals)

    tl.store(out_ptr + row_out_offset + out_pos, max_val, mask=out_mask)


def triton_maxpool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
    BLOCK_OUT: int = 128,
):
    """
    1D MaxPool implemented in Triton.

    Args:
        x (Tensor): (N, C, L_in) input tensor on CUDA.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    if stride is None:
        stride = kernel_size

    N, C, L_in = x.shape
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out = torch.empty((N, C, L_out), device=x.device, dtype=x.dtype)

    grid = (N * C, (L_out + BLOCK_OUT - 1) // BLOCK_OUT)
    maxpool1d_kernel[grid](
        x,
        out,
        N,
        C,
        L_in,
        stride,
        padding,
        L_out,
        BLOCK_OUT=BLOCK_OUT,
        KERNEL_SIZE=kernel_size,
        DILATION=dilation,
    )
    return out


class ModelNew(nn.Module):
    """
    MaxPool1D using a custom Triton kernel (no indices support).
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
    ):
        super().__init__()
        if return_indices:
            raise NotImplementedError("return_indices=True is not supported.")
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_device = x.device
        if not x.is_cuda:
            x = x.cuda()
        out = triton_maxpool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if orig_device.type != "cuda":
            out = out.to(orig_device)
        return out