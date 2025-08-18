import torch
import torch.nn as nn
import triton
import triton.language as tl

# Problem dimensions (can be changed as needed)
M = 4096
N = 4096


@triton.jit
def row_scale_kernel(
    a_ptr,                 # 1D tensor A (length N)
    b_ptr,                 # 2D tensor B (shape N x M)
    c_ptr,                 # 2D tensor C (output, shape N x M)
    N, M,                  # sizes
    stride_bn, stride_bm,  # B strides (col, row)
    stride_cn, stride_cm,  # C strides (col, row)
    BLOCK_M: tl.constexpr  # number of columns handled by a single program
):
    pid_row = tl.program_id(0)   # row index
    pid_col = tl.program_id(1)   # column-block index

    row_idx = pid_row
    col_start = pid_col * BLOCK_M
    offs_m = col_start + tl.arange(0, BLOCK_M)
    mask = offs_m < M

    # Load scaling factor for this row
    scale = tl.load(a_ptr + row_idx)

    # Compute pointers for B and C
    b_ptrs = b_ptr + row_idx * stride_bm + offs_m * stride_bn
    c_ptrs = c_ptr + row_idx * stride_cm + offs_m * stride_cn

    # Load, scale, and store
    vals = tl.load(b_ptrs, mask=mask, other=0.0)
    vals = vals * scale
    tl.store(c_ptrs, vals, mask=mask)


def triton_diag_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute diag(A) @ B using a custom Triton kernel.

    A: (N,)  – scaling vector
    B: (N, M) – matrix to be row-scaled
    Returns C = diag(A) @ B  (shape N x M)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    A_ = A.contiguous()
    B_ = B.contiguous()

    N, M = B_.shape
    C = torch.empty_like(B_)

    stride_bm, stride_bn = B_.stride()      # (row, col)
    stride_cm, stride_cn = C.stride()       # (row, col)

    BLOCK_M = 128
    grid = (N, triton.cdiv(M, BLOCK_M))

    row_scale_kernel[grid](
        A_,
        B_,
        C,
        N,
        M,
        stride_bn,
        stride_bm,
        stride_cn,
        stride_cm,
        BLOCK_M=BLOCK_M,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model replacing diag(A) @ B with a Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Fast CUDA path
        if A.is_cuda and B.is_cuda:
            return triton_diag_mm(A, B)
        # Fallback CPU/other device path (no Triton)
        return A.unsqueeze(1) * B


def get_inputs():
    # Generate random CUDA inputs for benchmarking
    A = torch.rand(N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]


def get_init_inputs():
    return []