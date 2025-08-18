import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _div_const_kernel(
    x_ptr,           # pointer to input tensor
    out_ptr,         # pointer to output tensor
    const,           # division constant
    n_elements,      # total number of elements
    BLOCK_SIZE: tl.constexpr,  # elements handled by each program
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x / const
    tl.store(out_ptr + offsets, y, mask=mask)


def _triton_div_const(x: torch.Tensor, const: float) -> torch.Tensor:
    """
    Element-wise division by a scalar using a Triton kernel.
    """
    assert x.is_cuda, "Input tensor must reside on GPU"
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _div_const_kernel[grid](x, out, const, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class _DivConstFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, const: float):
        ctx.const = const
        return _triton_div_const(x, const)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output / ctx.const, None


def div_const(x: torch.Tensor, const: float) -> torch.Tensor:
    """
    Autograd-compatible wrapper for Triton division.
    """
    return _DivConstFunction.apply(x, const)


class ModelNew(nn.Module):
    """
    Optimized model: keeps convolution & instance normalization from PyTorch,
    replaces the final element-wise division with a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = div_const(x, self.divide_by)
        return x


# ---- Helpers (unchanged from original code) ----
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
divide_by = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]