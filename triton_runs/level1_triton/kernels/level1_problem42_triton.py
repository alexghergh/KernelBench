import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,                    # *Pointer* to input tensor (B, C, H, W)
    out_ptr,                  # *Pointer* to output tensor (B, C, OH, OW)
    n_elements,               # Total number of output elements
    B, C, IH, IW, OH, OW,     # Tensor dimensions
    BLOCK_SIZE: tl.constexpr, # Number of threads per block
    K: tl.constexpr,          # Kernel size (assumed square)
    STRIDE: tl.constexpr,     # Stride
    PAD: tl.constexpr,        # Padding
    DIL: tl.constexpr,        # Dilation
):
    # Linear indices handled by this program
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Convert 1-D index into (b, c, oh, ow)
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    c  = tmp % C
    b  = tmp // C

    # Top-left corner of the receptive field in the input tensor
    in_row_origin = oh * STRIDE - PAD
    in_col_origin = ow * STRIDE - PAD

    # Initialise running max
    max_val = tl.full([BLOCK_SIZE], -1.0e30, dtype=tl.float32)

    # Iterate over the K×K window
    for kr in tl.static_range(K):
        row = in_row_origin + kr * DIL
        row_ok = (row >= 0) & (row < IH)

        for kc in tl.static_range(K):
            col   = in_col_origin + kc * DIL
            col_ok = (col >= 0) & (col < IW)

            curr_mask = row_ok & col_ok & mask

            # Flattened input index
            idx = ((b * C + c) * IH + row) * IW + col
            val = tl.load(x_ptr + idx, mask=curr_mask, other=-1.0e30)
            max_val = tl.where(val > max_val, val, max_val)

    # Store results
    tl.store(out_ptr + offs, max_val, mask=mask)


def triton_maxpool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    """
    Wrapper that launches the Triton max-pool kernel.
    Supports NCHW tensors on CUDA.
    """
    assert x.is_cuda and x.ndim == 4, "Input must be a CUDA tensor of shape (B, C, H, W)"
    B, C, IH, IW = x.shape

    K = kernel_size
    S = stride
    P = padding
    D = dilation

    # Output spatial dimensions (same formula PyTorch uses)
    OH = (IH + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (IW + 2 * P - D * (K - 1) - 1) // S + 1

    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
    n_elements = out.numel()

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    maxpool2d_kernel[grid](
        x, out,
        n_elements, B, C, IH, IW, OH, OW,
        BLOCK_SIZE=BLOCK_SIZE,
        K=K,
        STRIDE=S,
        PAD=P,
        DIL=D,
    )
    return out


class ModelNew(nn.Module):
    """
    Model that replaces nn.MaxPool2d with a custom Triton implementation.
    Falls back to torch.nn.functional.max_pool2d for non-CUDA tensors.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return triton_maxpool2d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
            )
        # CPU fallback
        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )