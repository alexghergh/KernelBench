import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        mask_b = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul_transposeA(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A.T @ B using a custom Triton GEMM kernel.
    A: shape (K, M)
    B: shape (K, N)
    Returns:
        C: shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must reside on CUDA device."
    K, M = A.shape
    K2, N = B.shape
    assert K == K2, "Dimension mismatch for matmul."

    A = A.contiguous()
    B = B.contiguous()

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(1), A.stride(0),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model computing C = A.T @ B via a custom Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul_transposeA(A, B)


# ------------------------------------------------------------------
# Helpers for benchmarking – kept identical to the original script
# ------------------------------------------------------------------
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


def get_inputs():
    A = torch.rand(K, M, device="cuda", dtype=torch.float32)
    B = torch.rand(K, N, device="cuda", dtype=torch.float32)
    return [A, B]


def get_init_inputs():
    return []