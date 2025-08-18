import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _4d_mm_kernel(
    A_ptr, B_ptr, C_ptr,                   # pointers to matrices
    M, N, K,                               # dimensions
    stride_am, stride_ak,                  # A strides
    stride_bk, stride_bn,                  # B strides
    stride_cm, stride_cn,                  # C strides
    BLOCK_M: tl.constexpr,                 # tile sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B
    A is (M, K), B is (K, N), C is (M, N)
    """
    pid_m = tl.program_id(0)               # program id along M dimension
    pid_n = tl.program_id(1)               # program id along N dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + stride_am * offs_m[:, None] + stride_ak * (k + offs_k)[None, :]
        b_ptrs = B_ptr + stride_bk * (k + offs_k)[:, None] + stride_bn * offs_n[None, :]

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    c_ptrs = C_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_bijl_lk(A: torch.Tensor, B: torch.Tensor,
                   BLOCK_M: int = 64, BLOCK_N: int = 128, BLOCK_K: int = 32):
    """
    Performs the einsum 'bijl,lk->bijk' using a custom Triton kernel.
    """
    assert A.ndim == 4, "A must be a 4D tensor"
    assert B.ndim == 2, "B must be a 2D tensor"
    assert A.shape[-1] == B.shape[0], "Mismatched inner dimensions"

    # Ensure tensors are on the same CUDA device and contiguous
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.to(A.device)

    A = A.contiguous()
    B = B.contiguous()

    b, i, j, l = A.shape
    k = B.shape[1]

    M = b * i * j
    N = k
    K = l

    # Flatten A for 2D matmul
    A_2d = A.view(M, K)
    B_2d = B
    C_2d = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Strides (in elements, not bytes)
    stride_am, stride_ak = A_2d.stride()
    stride_bk, stride_bn = B_2d.stride()
    stride_cm, stride_cn = C_2d.stride()

    # Launch configuration
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _4d_mm_kernel[grid](
        A_2d, B_2d, C_2d,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return C_2d.view(b, i, j, k)


class ModelNew(nn.Module):
    """
    Optimized version of the original Model using a custom Triton kernel
    for the 'bijl,lk->bijk' einsum.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_bijl_lk(A, B)