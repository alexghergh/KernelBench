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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using the custom Triton kernel.
    Expects A of shape (M, K) and B of shape (K, N), both on CUDA.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 tensors are supported."

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match."

    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32  # Suitable for the given small K

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return C


class ModelNew(nn.Module):
    """
    Optimized model that replaces torch.matmul with a custom Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


# Shape parameters (unchanged from original problem)
M = 16384 * 2
N = 16384 * 2
K = 32 * 2


def get_inputs():
    A = torch.rand(M, K, device="cuda", dtype=torch.float32)
    B = torch.rand(K, N, device="cuda", dtype=torch.float32)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed