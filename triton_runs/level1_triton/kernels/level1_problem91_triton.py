import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def flip_kernel(
    x_ptr,            # input pointer
    out_ptr,          # output pointer
    row_size,         # elements per row
    BLOCK_SIZE: tl.constexpr,
):
    # 2-D launch configuration: (rows, column_blocks)
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # Starting column index handled by this program
    col_start = block_id * BLOCK_SIZE
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size

    # Source positions (reversed)
    src_offsets = row_size - 1 - offsets

    src_ptrs = x_ptr + row_id * row_size + src_offsets
    dst_ptrs = out_ptr + row_id * row_size + offsets

    values = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, values, mask=mask)


def triton_flip(x: torch.Tensor, dim: int):
    """
    Fast flip (reverse) along `dim` implemented with a Triton kernel.
    Supports any tensor dimensionality; the target dimension is made
    contiguous via permutation before the kernel launch.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device")

    dim = dim if dim >= 0 else x.dim() + dim
    # Move the target dimension to the last position for contiguous access
    if dim != x.dim() - 1:
        perm = list(range(x.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_perm = x.permute(perm).contiguous()
        need_permute_back = True
    else:
        x_perm = x.contiguous()
        need_permute_back = False

    row_size = x_perm.shape[-1]
    n_rows = x_perm.numel() // row_size
    out_perm = torch.empty_like(x_perm)

    BLOCK_SIZE = 256

    grid = lambda meta: (
        n_rows,                          # one program per row
        triton.cdiv(row_size, meta["BLOCK_SIZE"])  # column blocks
    )

    flip_kernel[grid](x_perm, out_perm, row_size, BLOCK_SIZE=BLOCK_SIZE)

    if need_permute_back:
        perm = list(range(x.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        return out_perm.permute(perm)
    return out_perm


class ModelNew(nn.Module):
    """
    Optimized reverse cumulative sum using Triton-based flips to
    avoid costly data‐dependent indexing on the GPU.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x_rev = triton_flip(x, self.dim)
        y = torch.cumsum(x_rev, dim=self.dim)
        return triton_flip(y, self.dim)


# -----------------------------------------------------------------------
# Auxiliary helpers (mirroring the original script)
# -----------------------------------------------------------------------
batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    return [torch.rand(batch_size, *input_shape, device="cuda")]


def get_init_inputs():
    return [dim]