import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def clamp_div_kernel(
    x_ptr,          # Pointer to input
    out_ptr,        # Pointer to output
    min_val,        # Scalar clamp minimum
    inv_div,        # Scalar 1.0 / divisor
    n_elements,     # Number of elements to process
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x = tl.maximum(x, min_val)       # clamp
    x = x * inv_div                  # divide
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_clamp_div(x: torch.Tensor, min_value: float, divisor: float):
    """
    Fused clamp & divide implemented with a Triton kernel.
    Computes: y = max(x, min_value) / divisor
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    inv_div = 1.0 / divisor
    clamp_div_kernel[grid](
        x, out,
        min_value, inv_div,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses a Triton kernel to fuse clamp and division
    after a ConvTranspose3d operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding
        )
        self.min_value = float(min_value)
        self.divisor = float(divisor)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_clamp_div(x, self.min_value, self.divisor)
        return x