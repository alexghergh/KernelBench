import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_max_sub_mean_gelu_kernel(
    x_ptr,                 # *f32 [B, F]
    out_ptr,               # *f32 [B, 1]
    stride_x_row,          # int
    stride_x_col,          # int
    stride_out_row,        # int
    n_cols: tl.constexpr,  # F
    BLOCK_SIZE: tl.constexpr,
):
    """
    For each row:
        v = max(x, dim=1, keepdim=True)
        v = v - v.mean(dim=1, keepdim=True)   # always 0
        v = gelu(v)                           # gelu(0) = 0
    Returns a tensor of shape (B, 1)
    """
    row_id = tl.program_id(0)

    # Pointer to the start of this row
    row_ptr = x_ptr + row_id * stride_x_row

    # Compute max across the row in tiles of BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    row_max = tl.full((), -1e20, tl.float32)

    for col_start in tl.static_range(0, n_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < n_cols
        ptrs = row_ptr + cols * stride_x_col
        vals = tl.load(ptrs, mask=mask, other=-1e20)
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))

    # After reduction, the subsequent operations collapse to zeros,
    # but we keep them for semantic correctness.
    diff = row_max - row_max  # zero
    k = 0.7071067811865476    # 1/sqrt(2)
    gelu = 0.5 * diff * (1.0 + tl.math.erf(diff * k))

    # Store result
    tl.store(out_ptr + row_id * stride_out_row, gelu)


def triton_fused_op(x: torch.Tensor, max_dim: int) -> torch.Tensor:
    """
    Replaces:
        x = torch.max(x, dim=max_dim, keepdim=True).values
        x = x - x.mean(dim=1, keepdim=True)
        x = torch.nn.functional.gelu(x)
    Currently supports max_dim == 1 for 2-D tensors.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.dim() == 2 and max_dim == 1, "Only 2-D tensors with max_dim==1 supported."

    B, F = x.shape
    out = torch.empty(B, 1, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 1024  # Tunable
    grid = (B,)

    fused_max_sub_mean_gelu_kernel[grid](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        n_cols=F,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton to fuse max + mean subtraction + GELU.
    """
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        x = self.gemm(x)
        x = triton_fused_op(x, self.max_dim)
        return x


# --------------------------------------------------------------------
# Helper functions (unchanged API)
batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1


def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda")]


def get_init_inputs():
    return [in_features, out_features, max_dim]