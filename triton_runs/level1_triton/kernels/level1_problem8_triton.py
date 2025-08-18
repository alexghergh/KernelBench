import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,           # pointers to matrices
    M, N, K,                       # dimensions
    stride_am, stride_ak,          # strides for A
    stride_bk, stride_bn,          # strides for B
    stride_cm, stride_cn,          # strides for C
    BLOCK_M: tl.constexpr,         # tile sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    A simple yet efficient GEMM kernel: C[M, N] = A[M, K] @ B[K, N]
    """
    pid = tl.program_id(0)

    # Shape of the launch grid
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // grid_n                       # row tile index
    pid_n = pid % grid_n                        # col tile index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k[None, :] * stride_ak)
        b_ptrs = B_ptr + (k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper around the Triton GEMM kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"

    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Dimension mismatch"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda meta: (
        ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"])
        * ((N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]),
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
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
    Optimized model that replaces torch.matmul with a Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        return triton_matmul(A, B)


# Original shapes (for reference and input generation)
M = 8205
K = 2949
N = 5921


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]


def get_init_inputs():
    return []