import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tanh_kernel(
    x_ptr,          # pointer to input
    out_ptr,        # pointer to output
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,  # number of elements each program processes
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Numerically stable tanh:
    # tanh(x) = sign(x) * (1 - e) / (1 + e)  where e = exp(-2 * |x|)
    abs_x = tl.abs(x)
    e = tl.exp(-2.0 * abs_x)
    numerator = 1.0 - e
    denominator = 1.0 + e
    tanh_val = numerator / denominator
    sign = tl.where(x >= 0, 1.0, -1.0)
    y = sign * tanh_val

    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_tanh(x: torch.Tensor) -> torch.Tensor:
    """Applies a Triton-accelerated tanh to the input tensor."""
    assert x.is_cuda, "Input tensor must be on CUDA"
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    BLOCK_SIZE = 4096  # Tunable

    grid = lambda meta: (
        (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    tanh_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Tanh activation using a custom Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_tanh(x)


# -----------------------------------------------------------------------------
# Utilities for benchmarking / external usage (unchanged interface)
# -----------------------------------------------------------------------------
batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed