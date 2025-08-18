import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softsign_kernel(
    x_ptr,             # *float32 tensor
    out_ptr,           # *float32 tensor
    n_elements,        # int32
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise Softsign: y = x / (1 + |x|)
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x / (1.0 + tl.abs(x))
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_softsign(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Softsign activation using a custom Triton kernel.
    """
    assert x.is_cuda, "Input tensor must reside on CUDA device."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    softsign_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Softsign activation using Triton.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softsign(x)


# I/O helpers (kept identical to original interface)
batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda")
    return [x]


def get_init_inputs():
    return []