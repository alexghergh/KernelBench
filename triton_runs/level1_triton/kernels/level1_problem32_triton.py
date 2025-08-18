import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardtanh_kernel(
    x_ptr,           # * pointer to input
    out_ptr,         # * pointer to output
    n_elements,      # total number of elements
    min_val,         # lower clamp bound
    max_val,         # upper clamp bound
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x = tl.maximum(x, min_val)
    x = tl.minimum(x, max_val)

    tl.store(out_ptr + offsets, x, mask=mask)


def triton_hardtanh(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0):
    """
    Applies a HardTanh (clamp) using a custom Triton kernel.
    Falls back to torch.clamp for tensors on CPU.
    """
    if not x.is_cuda:
        return torch.clamp(x, min_val, max_val)

    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    hardtanh_kernel[grid](
        x, out,
        n_elements,
        min_val,
        max_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Model with Triton-accelerated HardTanh activation.
    """
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardtanh(x, self.min_val, self.max_val)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda")
    return [x]


def get_init_inputs():
    return []