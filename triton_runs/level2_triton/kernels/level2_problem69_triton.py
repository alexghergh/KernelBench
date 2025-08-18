import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_hs_relu_kernel(
    x_ptr,         # Pointer to input tensor
    out_ptr,       # Pointer to output tensor
    n_elements,    # Total number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute fused HardSwish + ReLU
    #   f(x) = 0,                 x <= 0
    #   f(x) = x*(x+3)/6,         0 < x < 3
    #   f(x) = x,                 x >= 3
    zero = tl.zeros_like(x)
    out = tl.where(x >= 3.0, x, zero)                     # x >= 3
    mid_mask = (x > 0.0) & (x < 3.0)                      # 0 < x < 3
    mid_val = x * (x + 3.0) / 6.0
    out = tl.where(mid_mask, mid_val, out)                # merge results

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_hardswish_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the fused HardSwish followed by ReLU using a Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_hs_relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps the convolution from PyTorch but replaces
    the HardSwish + ReLU sequence with a single fused Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = fused_hardswish_relu(x)
        return x


# -------------------------------------------------
# Helpers (unchanged API vs. original submission)
# -------------------------------------------------
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]