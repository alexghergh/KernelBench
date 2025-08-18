import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Constants used for input generation (kept identical to the original code)
M = 16384 * 4
N = 4096 * 4


@triton.jit
def mul_scalar_kernel(
    a_ptr,        # Pointer to the input matrix A
    scalar,       # Scalar multiplier (float32)
    out_ptr,      # Pointer to the output matrix C
    n_elements,   # Total number of elements in A / C
    BLOCK_SIZE: tl.constexpr,
):
    # Identify which block of elements this program instance will handle
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load, multiply by scalar, and store
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    c = a * scalar
    tl.store(out_ptr + offsets, c, mask=mask)


def triton_mul_scalar(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Multiply tensor `A` by scalar `s` using a custom Triton kernel.
    """
    if not A.is_cuda:
        raise RuntimeError("Tensor A must reside on a CUDA device.")

    A = A.contiguous()
    out = torch.empty_like(A)

    n_elems = A.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elems + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Ensure that the scalar has the same dtype as A
    scalar = A.new_full((), s).item()

    mul_scalar_kernel[grid](A, scalar, out, n_elems, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-scalar multiplication (C = A * s)
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        # Move to GPU if necessary
        original_device = A.device
        if not A.is_cuda:
            A_cuda = A.cuda()
        else:
            A_cuda = A

        # Run the Triton kernel
        C_cuda = triton_mul_scalar(A_cuda, s)

        # Move back to original device if needed
        if original_device.type == "cpu":
            return C_cuda.cpu()
        return C_cuda


def get_inputs():
    A = torch.rand(M, N)  # CPU tensor by default
    s = 3.14
    return [A, s]


def get_init_inputs():
    return []  # No special initialization inputs needed