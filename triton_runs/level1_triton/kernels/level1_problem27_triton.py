import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def selu_kernel(
    x_ptr,          # Input tensor
    out_ptr,        # Output tensor
    n_elements,     # Number of elements to process
    alpha,          # SELU α
    scale,          # SELU λ
    BLOCK_SIZE: tl.constexpr,  # Number of elements processed by a single program
):
    # Program id identifies the block of data this instance will process
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)

    # SELU activation: scale * (x if x > 0 else alpha * (exp(x) - 1))
    out = tl.where(x > 0, scale * x, scale * alpha * (tl.exp(x) - 1))

    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_selu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies SELU activation using a custom Triton kernel.
    """
    if not x.is_cuda:
        # Fallback to PyTorch implementation if tensor is on CPU
        return torch.nn.functional.selu(x)

    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable – good default for large tensors

    # SELU constants
    alpha = 1.6732632423543772
    scale = 1.0507009873554805

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    selu_kernel[grid](x, out, n_elements, alpha, scale, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs SELU activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_selu(x)


# -----------------------------------------------------------------------------
# Helper functions to create inputs identical to the original architecture
# -----------------------------------------------------------------------------
batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda")
    return [x]


def get_init_inputs():
    return []  # No initialization tensors required