import torch
import torch.nn as nn
import triton
import triton.language as tl

# Matrix size used by get_inputs (can be changed freely by the caller)
N = 4096


@triton.jit
def _triu_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_Am, stride_Ak,
    stride_Bk, stride_Bn,
    stride_Cm, stride_Cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel computing C = triu(A @ B) for square upper-triangular matrices.
    Only the upper-triangular part of C is written back to memory.
    """
    pid_m = tl.program_id(0)  # program index along M dimension
    pid_n = tl.program_id(1)  # program index along N dimension

    row_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over K dimension
    for k in range(0, N, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = A_ptr + row_offsets[:, None] * stride_Am + k_offsets[None, :] * stride_Ak
        b_ptrs = B_ptr + k_offsets[:, None] * stride_Bk + col_offsets[None, :] * stride_Bn

        mask_a = (row_offsets[:, None] < N) & (k_offsets[None, :] < N)
        mask_b = (k_offsets[:, None] < N) & (col_offsets[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + row_offsets[:, None] * stride_Cm + col_offsets[None, :] * stride_Cn
    store_mask = (
        (row_offsets[:, None] < N)
        & (col_offsets[None, :] < N)
        & (row_offsets[:, None] <= col_offsets[None, :])  # keep only upper triangle
    )
    tl.store(c_ptrs, acc, mask=store_mask)


def triton_triu_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """
    Compute triu(A @ B) with a custom Triton kernel.
    Assumes A and B are square, on CUDA, and of the same shape/dtype (float32).
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.shape == B.shape and A.ndim == 2 and A.shape[0] == A.shape[1], "Inputs must be square matrices of equal size."
    assert A.dtype == torch.float32, "Only float32 is supported in this kernel."

    N = A.shape[0]
    C = torch.zeros_like(A)

    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    _triu_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that multiplies two upper-triangular matrices using a custom Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and contiguity
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        A = A.contiguous()
        B = B.contiguous()

        return triton_triu_matmul(A, B)


def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: Two upper triangular CUDA tensors of shape (N, N).
    """
    A = torch.triu(torch.rand(N, N, device="cuda", dtype=torch.float32))
    B = torch.triu(torch.rand(N, N, device="cuda", dtype=torch.float32))
    return [A, B]


def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []