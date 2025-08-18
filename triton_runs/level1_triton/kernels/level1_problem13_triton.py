import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,                              # pointers to matrices
    M, N, K,                                          # dimensions
    stride_am, stride_ak,                            # strides for A
    stride_bk, stride_bn,                            # strides for B
    stride_cm, stride_cn,                            # strides for C
    BLOCK_M: tl.constexpr,                           # tile sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes C = A @ B
      A: (M, K) row-major
      B: (K, N) row-major
      C: (M, N) row-major
    """

    pid_m = tl.program_id(0)          # program id along M dimension
    pid_n = tl.program_id(1)          # program id along N dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # pointers for first K-tile
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & ((offs_k[None, :] + k) < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=((offs_k[:, None] + k) < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Launches the custom Triton kernel to compute matrix multiplication.
    Supports float32 tensors stored in row-major (contiguous) layout.
    """

    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible matrix dimensions for matmul"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Strides (row-major / C-contiguous)
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return C


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return triton_matmul(A, B)