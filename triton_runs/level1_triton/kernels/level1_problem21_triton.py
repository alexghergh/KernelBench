import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
    x_ptr,          # * Pointer to input data
    out_ptr,        # * Pointer to output data
    n_elements,     # * Number of elements to process
    BLOCK_SIZE: tl.constexpr,  # * How many elements this program instance will process
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Applies element-wise sigmoid to `x` using a Triton kernel.
    Falls back to `torch.sigmoid` if `x` is not on CUDA.
    """
    if not x.is_cuda:
        return torch.sigmoid(x)

    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    sigmoid_kernel[grid](
        x, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom Triton kernel for Sigmoid on CUDA tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sigmoid(x)


# ---- Helpers for benchmarking / integration (unchanged) ---- #
batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed