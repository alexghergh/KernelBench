import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _min_reduce_last_dim_kernel(
    x_ptr,              # * pointer to input matrix [ROWS, K] *
    out_ptr,            # * pointer to output vector [ROWS] *
    stride_x_row,       # * stride between rows (in elements) *
    stride_x_col,       # * stride between cols (in elements, usually 1) *
    K,                  # * size of the reduction dimension *
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)                      # each program handles one row
    offs_k = tl.arange(0, BLOCK_K)              # vector of column indices
    ptrs = x_ptr + pid * stride_x_row + offs_k * stride_x_col
    mask = offs_k < K
    INF = float("inf")
    acc = tl.load(ptrs, mask=mask, other=INF)   # initialise accumulator

    for k in range(BLOCK_K, K, BLOCK_K):
        ptrs = x_ptr + pid * stride_x_row + (offs_k + k) * stride_x_col
        mask = (offs_k + k) < K
        vals = tl.load(ptrs, mask=mask, other=INF)
        acc = tl.minimum(acc, vals)             # running minimum

    row_min = tl.min(acc, axis=0)               # reduce vector to scalar
    tl.store(out_ptr + pid, row_min)            # write result


def _triton_min_last_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the minimum over the last dimension of `x` using the Triton kernel.
    The input must be contiguous, on CUDA, and of dtype float32.
    """
    assert x.is_cuda, "Input must be on CUDA."
    assert x.is_contiguous(), "Input must be contiguous."
    assert x.dtype == torch.float32, "Only float32 is supported by the Triton kernel."

    K = x.shape[-1]
    rows = x.numel() // K
    x_2d = x.view(rows, K)

    out = torch.empty(rows, device=x.device, dtype=x.dtype)

    BLOCK_K = 1024
    grid = lambda meta: (rows,)

    _min_reduce_last_dim_kernel[grid](
        x_2d,
        out,
        x_2d.stride(0),
        x_2d.stride(1),
        K,
        BLOCK_K=BLOCK_K,
    )

    return out.view(*x.shape[:-1])


def triton_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Wrapper that applies the Triton min-reduction kernel over an arbitrary dimension.
    Falls back to PyTorch if the tensor is not suitable for Triton.
    """
    if dim < 0:
        dim += x.dim()

    if dim == x.dim() - 1:
        return _triton_min_last_dim(x.contiguous())

    # Move the reduction dimension to the last position
    x_moved = x.movedim(dim, -1).contiguous()
    reduced = _triton_min_last_dim(x_moved)
    return reduced


class ModelNew(nn.Module):
    """
    Optimized model that performs a min reduction over a specified dimension using
    a custom Triton kernel for improved performance.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32:
            return triton_min(x, self.dim)
        # Fallback for CPUs or unsupported dtypes
        return torch.min(x, dim=self.dim)[0]