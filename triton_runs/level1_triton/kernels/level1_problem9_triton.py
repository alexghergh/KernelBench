import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,          # pointers to matrices
    M, N, K,                      # dimensions
    stride_am, stride_ak,         # A strides
    stride_bk, stride_bn,         # B strides
    stride_cm, stride_cn,         # C strides
    BLOCK_SIZE_M: tl.constexpr,   # block sizes (tunable)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,   # number of row blocks handled by a program group
):
    # 2-D program id
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Starting indices for this program
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    # Offsets
    offs_m = row_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = col_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_SIZE_K):
        k_idxs = k0 + offs_k

        # Pointers for the current tile of A and B
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_idxs[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks to guard memory accesses
        a_mask = (offs_m[:, None] < M) & (k_idxs[None, :] < K)
        b_mask = (k_idxs[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Compute and accumulate
        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Thin wrapper that dispatches the Triton kernel.
    Works for arbitrary strides on A, B (must share K dimension).
    """
    # Move to CUDA if necessary and ensure contiguous storage for performance
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match for matmul"

    # Output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Kernel launch parameters (can be tuned)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M  = 8

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for tall-skinny / skinny-tall GEMM.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)