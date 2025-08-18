import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,         # Pointers to matrices
    M, N, K,                     # Dimensions
    stride_am, stride_ak,        # Strides for A
    stride_bk, stride_bn,        # Strides for B
    stride_cm, stride_cn,        # Strides for C
    BLOCK_M: tl.constexpr,       # Tile size in M dimension
    BLOCK_N: tl.constexpr,       # Tile size in N dimension
    BLOCK_K: tl.constexpr        # Tile size in K dimension
):
    # Program IDs for 2-D launch grid
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for the tile this program will compute
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first blocks of A and B
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension
    k_tiles = tl.cdiv(K, BLOCK_K)
    for _ in range(k_tiles):
        a_mask = (offs_m[:, None] < M) & ((offs_k[None, :] + _ * BLOCK_K) < K)
        b_mask = ((offs_k[:, None] + _ * BLOCK_K) < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        # Advance block pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write back the results
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor,
                  BLOCK_M: int = 128,
                  BLOCK_N: int = 128,
                  BLOCK_K: int = 32) -> torch.Tensor:
    """
    Launches the Triton matrix multiplication kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Incompatible matrix shapes"

    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Strides (number of elements to jump to go to next row/col)
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    grid = ((M + BLOCK_M - 1) // BLOCK_M,
            (N + BLOCK_N - 1) // BLOCK_N)

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
    Optimized model that performs matrix multiplication with a
    custom Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


# Problem-specific dimensions
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


def get_inputs():
    A = torch.rand(M, K, device='cuda')
    B = torch.rand(K, N, device='cuda')
    return [A, B]


def get_init_inputs():
    return []