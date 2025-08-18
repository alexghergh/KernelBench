import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matvec_kernel(
    A_ptr,  # pointer to A [M, K]
    B_ptr,  # pointer to B [K]
    C_ptr,  # pointer to C [M]
    M,      # rows in A / C
    K,      # cols in A / len(B)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)                       # program id == row block id
    m_start = pid * BLOCK_SIZE_M                 # first row this program will compute
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_SIZE_K):
        kk = k0 + offs_k                                              # [BLOCK_SIZE_K]

        # load BLOCK_SIZE_M x BLOCK_SIZE_K tile from A
        a_ptrs = A_ptr + offs_m[:, None] * K + kk[None, :]
        a_mask = (offs_m[:, None] < M) & (kk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)                   # [M_block, K_block]

        # load BLOCK_SIZE_K elements from B
        b = tl.load(B_ptr + kk, mask=kk < K, other=0.0)               # [K_block]
        b = tl.reshape(b, (1, BLOCK_SIZE_K))                          # broadcast for multiply

        acc += tl.sum(a * b, axis=1)                                  # dot-product for each row

    # store results
    c_ptrs = C_ptr + offs_m
    tl.store(c_ptrs, acc, mask=offs_m < M)


def triton_matvec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A @ B (matrix–vector product) using a custom Triton kernel.
    Shapes:
      A : [M, K]
      B : [K] or [K, 1]
      C : [M, 1]
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
    assert A.ndim == 2, "A must be 2-D"
    M, K = A.shape
    assert B.shape in {(K,), (K, 1)}, "B must have shape (K,) or (K,1)"

    A = A.contiguous()
    B = B.contiguous().view(K)                     # make 1-D
    C = torch.empty(M, device=A.device, dtype=A.dtype)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 128

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)

    matvec_kernel[grid](
        A, B, C,
        M, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return C.view(M, 1)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using a Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matvec(A, B)


# Constants used to generate sample inputs
M = 256 * 8          # 2048
K = 131072 * 8       # 1,048,576


def get_inputs():
    A = torch.rand(M, K, device='cuda')
    B = torch.rand(K, 1, device='cuda')
    return [A, B]


def get_init_inputs():
    return []