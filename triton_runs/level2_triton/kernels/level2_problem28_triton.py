import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------
# Triton matrix multiplication
# --------------------------
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak)
        b_ptrs = B_ptr + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn)

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

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    x      : [M, K]
    weight : [N, K]   (row-major)
    bias   : [N] or None
    returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda, "Input and weight must be CUDA tensors"
    x = x.contiguous()
    weight_t = weight.t().contiguous()  # [K, N]

    M, K = x.shape
    K_w, N = weight_t.shape
    assert K == K_w, "Incompatible shapes between input and weight"

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    _matmul_kernel[grid](
        x, weight_t, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )

    if bias is not None:
        out += bias

    return out


# --------------------------
# Torch Module wrapping the Triton linear
# --------------------------
class TritonLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        return _triton_linear(x, self.weight, self.bias)


# --------------------------
# Optimised Model
# --------------------------
class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication with a custom Triton kernel,
    instance normalization, summation, residual addition, and multiplication.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bmm = TritonLinear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, x, y):
        x = self.bmm(x)  # Custom Triton GEMM (+bias)
        x = self.instance_norm(x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
        x = x + y
        x = x * y
        return x