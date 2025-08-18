import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_kernel(
    x_ptr,         # *fp32
    out_ptr,       # *fp32
    scale,         # fp32 scalar
    n_elements,    # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * scale, mask=mask)


def triton_scale(x: torch.Tensor, scale: float, block_size: int = 1024):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    scale_kernel[grid](x, out, scale, n_elements, BLOCK_SIZE=block_size)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a convolution, scales the output using a custom Triton
    kernel, and then applies a channel-wise minimum.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = triton_scale(x, self.scale_factor)
        x = torch.min(x, dim=1, keepdim=True)[0]
        return x


# --------------------------------------------------------------------------
# Helper functions to create inputs (same signatures as the original code)
# --------------------------------------------------------------------------
batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]