import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr,  # (M, K)
    B_ptr,  # (K, N)
    C_ptr,  # (M, N)
    M,      # rows of A / C
    N,      # cols of B / C
    K,      # shared dim
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_3d_tensor_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs the same operation as torch.matmul for a 3D tensor A (N, M, K)
    and a 2D matrix B (K, L), but executes the core GEMM using a custom Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.dtype == B.dtype, "Input dtypes must match"
    N, M, K = A.shape
    K_b, L = B.shape
    assert K == K_b, "Incompatible shapes: {} vs {}".format(K, K_b)

    # Flatten leading dimensions so we launch a single large GEMM
    A_2d = A.reshape(-1, K).contiguous()        # (N*M, K)
    B_contig = B.contiguous()                   # (K, L)
    M_tot = A_2d.shape[0]

    C = torch.empty(M_tot, L, device=A.device, dtype=A.dtype)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (M_tot + BLOCK_M - 1) // BLOCK_M,
        (L + BLOCK_N - 1) // BLOCK_N,
    )

    batched_matmul_kernel[grid](
        A_2d, B_contig, C,
        M_tot, L, K,
        A_2d.stride(0), A_2d.stride(1),
        B_contig.stride(0), B_contig.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return C.reshape(N, M, L)


class ModelNew(nn.Module):
    """
    Optimized model that replaces torch.matmul with a custom Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_3d_tensor_matmul(A, B)


# Helper functions to generate example inputs on CUDA
N = 16
M = 1024
K = 2048
L = 768


def get_inputs():
    A = torch.rand(N, M, K, device="cuda")
    B = torch.rand(K, L, device="cuda")
    return [A, B]


def get_init_inputs():
    return []