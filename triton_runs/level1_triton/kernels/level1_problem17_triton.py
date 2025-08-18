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
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create ranges for the program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_indices = k0 + offs_k

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_indices[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_indices[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_indices[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_indices[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using a custom Triton kernel.
    A: (M, K)
    B: (N, K)
    Returns C: (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only fp32 supported"

    M, K = A.shape
    N, Kb = B.shape
    assert K == Kb, "Dimension mismatch"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),          # A strides
        B.stride(1), B.stride(0),          # Treat B as transposed
        C.stride(0), C.stride(1),          # C strides
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton to compute C = A @ B.T.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.cuda().contiguous()
        B = B.cuda().contiguous()
        return triton_matmul(A, B)


# Problem dimensions (kept identical to the original model)
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


def get_inputs():
    A = torch.rand(M, K, device="cuda", dtype=torch.float32)
    B = torch.rand(N, K, device="cuda", dtype=torch.float32)
    return [A, B]


def get_init_inputs():
    return []