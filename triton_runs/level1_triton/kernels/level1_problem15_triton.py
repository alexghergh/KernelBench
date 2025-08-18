import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tril_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,                          # matrix dimension (square)
    stride_am, stride_ak,       # A strides
    stride_bk, stride_bn,       # B strides
    stride_cm, stride_cn,       # C strides
    BLOCK_M: tl.constexpr,      # tile sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)        # block index on M dimension
    pid_n = tl.program_id(1)        # block index on N dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, N, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for the current tile
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for triangular property and bounds
        a_mask = (
            (offs_m[:, None] < N)
            & (offs_k[None, :] < N)
            & (offs_k[None, :] <= offs_m[:, None])   # A is lower triangular
        )
        b_mask = (
            (offs_k[:, None] < N)
            & (offs_n[None, :] < N)
            & (offs_n[None, :] <= offs_k[:, None])   # B is lower triangular
        )

        A = tl.load(A_ptrs, mask=a_mask, other=0.0)
        B = tl.load(B_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(A, B)

    # Only store lower–triangular part of the result
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (
        (offs_m[:, None] < N)
        & (offs_n[None, :] < N)
        & (offs_m[:, None] >= offs_n[None, :])       # keep lower triangle
    )
    tl.store(C_ptrs, acc, mask=c_mask)


def tril_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """
    Multiply two lower–triangular square matrices on GPU using a Triton kernel
    and return the lower–triangular result.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "float32 only."
    assert A.shape == B.shape and A.ndim == 2 and A.shape[0] == A.shape[1], \
        "A and B must be square matrices of the same shape."

    N = A.shape[0]
    C = torch.zeros_like(A)   # initialize output with zeros for upper triangle

    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))

    tril_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that multiplies two lower–triangular matrices using a custom
    Triton kernel and returns the lower–triangular result.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        # Ensure inputs are on GPU and contiguous
        A = A.contiguous().cuda()
        B = B.contiguous().cuda()
        return tril_matmul(A, B)


# -------------------------- helper functions -------------------------- #
M = 4096


def get_inputs():
    A = torch.rand(M, M, device="cuda", dtype=torch.float32)
    B = torch.rand(M, M, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization required