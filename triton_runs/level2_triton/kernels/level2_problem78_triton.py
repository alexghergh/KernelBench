import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def channel_sum_kernel(
    x_ptr,          # *fp32, flattened tensor (rows, C)
    out_ptr,        # *fp32, output tensor (rows,)
    C,              # number of channels
    BLOCK_SIZE: tl.constexpr,  # number of values each program will load
):
    pid = tl.program_id(0)                         # program index == row index
    offs = tl.arange(0, BLOCK_SIZE)                # [0, 1, 2, ... BLOCK_SIZE-1]
    ptrs = x_ptr + pid * C + offs                  # add proper stride
    mask = offs < C                                # mask out-of-bounds
    vals = tl.load(ptrs, mask=mask, other=0.0)     # load
    res = tl.sum(vals, axis=0)                     # reduction along the vector
    tl.store(out_ptr + pid, res)                   # write back


def triton_channel_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to `torch.sum(x, dim=1, keepdim=True)` for a 5-D tensor
    (N, C, D, H, W), but executed by a Triton kernel.
    """
    assert x.is_cuda, "Input must be on CUDA device"
    N, C, D, H, W = x.shape

    # Move channels to the last dimension so that they're contiguous
    x_contig = x.permute(0, 2, 3, 4, 1).contiguous()  # (N, D, H, W, C)
    rows = N * D * H * W
    x_flat = x_contig.view(rows, C)                   # (rows, C)

    # Prepare output
    out_flat = torch.empty(rows, device=x.device, dtype=x.dtype)

    # Choose BLOCK_SIZE >= C (power-of-two for best performance)
    if   C <= 32:   BLOCK_SIZE = 32
    elif C <= 64:   BLOCK_SIZE = 64
    elif C <= 128:  BLOCK_SIZE = 128
    elif C <= 256:  BLOCK_SIZE = 256
    elif C <= 512:  BLOCK_SIZE = 512
    else:           BLOCK_SIZE = 1024  # fallback (C must be <= 1024)

    grid = lambda meta: (rows,)

    channel_sum_kernel[grid](x_flat, out_flat, C, BLOCK_SIZE=BLOCK_SIZE)

    out = out_flat.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3).contiguous()  # (N,1,D,H,W)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that replaces the channel-wise sum with a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = triton_channel_sum(x)  # custom Triton reduction
        return x


# ----------------------------------------------------------------------
# Helper functions (same API as original code)
# ----------------------------------------------------------------------
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device='cuda')]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]