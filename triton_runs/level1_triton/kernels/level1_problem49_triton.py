import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------------------------
# Triton kernel: row-wise max
# ---------------------------
@triton.jit
def rowwise_max_kernel(
    x_ptr,      # pointer to input  (N, K) – row major, contiguous
    out_ptr,    # pointer to output (N,)
    N,          # number of rows
    K,          # reduction length
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)                 # row index
    row_ptr = x_ptr + pid * K              # pointer to the first element of the row

    acc = tl.full((), -float("inf"), dtype=tl.float32)  # accumulator (scalar)

    off = tl.arange(0, BLOCK_SIZE)         # static range [0, BLOCK_SIZE)

    k = 0
    while k < K:
        offs = k + off                     # element offsets for this iteration
        mask = offs < K                    # mask to stay within bounds
        vals = tl.load(row_ptr + offs, mask=mask, other=-float("inf"))
        vals = vals.to(tl.float32)         # widen for numeric stability
        acc = tl.maximum(acc, tl.max(vals, axis=0))
        k += BLOCK_SIZE

    tl.store(out_ptr + pid, acc)           # automatic cast to out_ptr dtype


# ---------------------------
# Python wrapper
# ---------------------------
def triton_max(x: torch.Tensor, dim: int):
    """
    Max-reduce `x` along `dim` using the custom Triton kernel.
    Only the values are returned (no indices).
    """
    assert x.is_cuda, "Input tensor must reside on CUDA device"

    if dim < 0:
        dim += x.ndim

    # Move the reduction axis to the last dimension to get contiguous slices
    perm = [d for d in range(x.ndim) if d != dim] + [dim]
    x_perm = x.permute(perm).contiguous()

    # Flatten to 2-D: (N, K) where K is the reduction length
    K = x_perm.size(-1)
    N = x_perm.numel() // K
    x_2d = x_perm.view(N, K)

    # Prepare output buffer
    out = torch.empty(N, device=x.device, dtype=x.dtype)

    # Launch Triton kernel
    BLOCK_SIZE = 1024  # Tunable
    grid = lambda meta: (N,)
    rowwise_max_kernel[grid](x_2d, out, N, K, BLOCK_SIZE=BLOCK_SIZE)

    # Reshape back (without the reduced dimension)
    out_shape = [x.size(d) for d in range(x.ndim) if d != dim]
    return out.view(*out_shape)


# ---------------------------
# Optimized model
# ---------------------------
class ModelNew(nn.Module):
    """
    Model that performs max-reduction along a chosen dimension
    using a high-performance Triton kernel.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max(x, self.dim)


# ---------------------------
# I/O helpers (unchanged API)
# ---------------------------
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device="cuda")
    return [x]

def get_init_inputs():
    return [1]  # default reduction dimension