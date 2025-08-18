import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def elu_kernel(
    x_ptr,          # *Pointer* to input
    out_ptr,        # *Pointer* to output
    n_elements,     # Number of elements to process
    alpha,          # ELU alpha
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)                           # Program ID
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                      # Bounds check
    x = tl.load(x_ptr + offsets, mask=mask, other=0) # Load
    out = tl.where(x > 0, x, alpha * (tl.exp(x) - 1))
    tl.store(out_ptr + offsets, out, mask=mask)      # Store


def triton_elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    ELU implemented with Triton. Falls back to torch.nn.functional.elu
    when the tensor is not on CUDA.
    """
    if not x.is_cuda:
        return F.elu(x, alpha=alpha)

    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    # Define grid
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    elu_kernel[grid](x, out, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom Triton kernel for ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_elu(x, self.alpha)


# ---------------------------------------------------------------------
# Helpers for the benchmark harness
# ---------------------------------------------------------------------
batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda")
    return [x]


def get_init_inputs():
    return [1.0]  # alpha