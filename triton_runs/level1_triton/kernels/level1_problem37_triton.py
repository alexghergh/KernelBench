import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------
#  Kernel 1: compute partial sums of squares for Frobenius norm
# ---------------------------------------------------------------------
@triton.jit
def sum_sq_kernel(
    x_ptr,           # *pointer* to data
    partial_ptr,     # *pointer* to per-block partial results
    n_elements,      # total number of elements in `x`
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sq = x * x
    acc = tl.sum(sq, 0)            # reduce within the block
    tl.store(partial_ptr + pid, acc)


# ---------------------------------------------------------------------
#  Kernel 2: scale the tensor with the inverse Frobenius norm
# ---------------------------------------------------------------------
@triton.jit
def scale_kernel(
    x_ptr,           # *pointer* to input tensor
    out_ptr,         # *pointer* to output tensor
    scale_ptr,       # *pointer* to scalar  (1 / ||x||_F)
    n_elements,      # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)     # broadcasts to all threads
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------
#  High-level wrapper around the two kernels
# ---------------------------------------------------------------------
def fro_norm_triton(x: torch.Tensor, *, block_size: int = 4096) -> torch.Tensor:
    """
    Computes x / ||x||_F using two Triton kernels.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()
    n_elements = x.numel()
    dtype = x.dtype

    # ------------------------------------------------------------
    # Pass 1 : partial reductions (sum of squares)
    # ------------------------------------------------------------
    num_blocks = (n_elements + block_size - 1) // block_size
    partial = torch.empty(num_blocks, dtype=torch.float32, device=x.device)

    sum_sq_kernel[(num_blocks,)](
        x,
        partial,
        n_elements,
        BLOCK_SIZE=block_size,
    )

    # Host reduction
    total_sum = partial.sum()
    norm = torch.sqrt(total_sum)
    inv_norm = (1.0 / norm).to(dtype)

    # ------------------------------------------------------------
    # Pass 2 : scale the tensor
    # ------------------------------------------------------------
    out = torch.empty_like(x)
    scale_tensor = torch.tensor(inv_norm, dtype=dtype, device=x.device)

    scale_kernel[(num_blocks,)](
        x,
        out,
        scale_tensor,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return out


# ---------------------------------------------------------------------
#  Optimized model
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Frobenius norm normalization using custom Triton kernels.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fro_norm_triton(x)