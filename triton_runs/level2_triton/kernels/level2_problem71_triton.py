import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def div_leakyrelu_kernel(
    x_ptr,            # input tensor
    out_ptr,          # output tensor
    n_elements,       # number of elements
    divisor,          # scalar divisor
    negative_slope,   # scalar negative slope
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = x / divisor
    out = tl.where(y >= 0, y, y * negative_slope)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_div_leakyrelu(
    x: torch.Tensor,
    divisor: float,
    negative_slope: float = 0.01,
    block_size: int = 1024,
):
    """
    Fuses division by a constant and LeakyReLU into a single Triton kernel.
    """
    assert x.is_cuda, "Input tensor must reside on GPU"

    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    div_leakyrelu_kernel[grid](
        x,
        out,
        n_elements,
        divisor,
        negative_slope,
        BLOCK_SIZE=block_size,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
    1. Standard 2D convolution (handled by cuDNN).
    2. Division by a constant fused with LeakyReLU via a custom Triton kernel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        divisor: float,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)
        self.negative_slope = float(negative_slope)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = triton_div_leakyrelu(x, self.divisor, self.negative_slope)
        return x


# ----------------------------------------------------------------------
# Helper functions (kept identical to the original interface)
# ----------------------------------------------------------------------
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]