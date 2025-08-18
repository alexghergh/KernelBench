import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,                # pointers to matrices
    M, N, K,                            # dimensions
    stride_am, stride_ak,               # strides for A
    stride_bk, stride_bn,               # strides for B
    stride_cm, stride_cn,               # strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ------------------------
    # Program ID calculations
    # ------------------------
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_id = pid // (GROUP_SIZE_M * num_pid_n)
    first_pid_m = group_id * GROUP_SIZE_M
    pid_m = first_pid_m + (pid // num_pid_n) % GROUP_SIZE_M
    pid_n = pid % num_pid_n

    # ------------------------
    # Create index ranges for the tile
    # ------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # pointers for A and B
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------------------
    # Loop over K dimension
    # ------------------------
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # ------------------------
    # Write back the result
    # ------------------------
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        C_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication C = A @ B using a custom Triton kernel.
    Supports float32 inputs stored in row-major (contiguous) layout.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 supported."

    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Inner dimensions must agree."

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        GROUP_SIZE_M=8,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs square matrix multiplication using a Triton kernel.
    """
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


# Input size (square matrices)
N = 2048 * 2


def get_inputs():
    A = torch.rand(N, N, device="cuda", dtype=torch.float32)
    B = torch.rand(N, N, device="cuda", dtype=torch.float32)
    return [A, B]


def get_init_inputs():
    return []