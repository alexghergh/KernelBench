import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,          # *pointer* to input
    out_ptr,        # *pointer* to output
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,  # block size (power-of-two is recommended)
):
    # Program id
    pid = tl.program_id(0)
    # Offset for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to guard memory accesses beyond the size of the tensor
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Constants
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies GELU to `x` using a custom Triton kernel.
    """
    assert x.is_cuda, "triton_gelu: input must be a CUDA tensor"
    x = x.contiguous()

    # Allocate output
    out = torch.empty_like(x)

    # Kernel launch parameters
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable; good default for most GPUs
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that uses a Triton-accelerated GELU activation.
    Falls back to torch.nn.functional.gelu when the input is on CPU.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return triton_gelu(x)
        # Fallback for CPU tensors
        return F.gelu(x)


# ----- Helper functions (kept identical to the original API) -----
batch_size = 4096
dim = 393216


def get_inputs():
    # Ensure the tensor is created on GPU if available, otherwise on CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(batch_size, dim, device=device)
    return [x]


def get_init_inputs():
    return []  # No special initialization required