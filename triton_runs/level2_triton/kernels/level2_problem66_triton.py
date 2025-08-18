import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_bias_kernel(
    A_ptr,          # (M, K) input
    B_ptr,          # (K, N) weight^T  (note the transpose!)
    Bias_ptr,       # (N,)   bias
    C_ptr,          # (M, N) output
    M, N, K,        # dimensions
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B + bias
    A: (M, K)
    B: (K, N)
    bias: (N,)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pointers for the first tile
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        b = tl.load(
            b_ptrs,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Write back
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_linear(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x        : (M, K)
    weight_t : (K, N)  -> transposed weight
    bias     : (N,)
    returns  : (M, N)
    """
    assert x.is_cuda and weight_t.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    x = x.contiguous()
    weight_t = weight_t.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    K_w, N = weight_t.shape
    assert K == K_w, "Incompatible shapes between input and weight"
    assert bias.numel() == N, "Bias must have shape (N,)"

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_bias_kernel[grid](
        x,
        weight_t,
        bias,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight_t.stride(0),
        weight_t.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a Triton kernel for the Linear layer.
    Dropout and Softmax remain in PyTorch.
    """

    def __init__(self, in_features: int, out_features: int, dropout_p: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters replicate nn.Linear layout
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

        self.dropout = nn.Dropout(dropout_p)

    def reset_parameters(self):
        # Same initialization strategy as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure parameters are on the same device as x
        if x.device != self.weight.device:
            self.weight.data = self.weight.data.to(x.device)
            self.bias.data = self.bias.data.to(x.device)

        # Triton-accelerated linear projection
        out = triton_linear(x, self.weight.t().contiguous(), self.bias)
        # Dropout + Softmax
        out = self.dropout(out)
        out = torch.softmax(out, dim=1)
        return out


# Pre-defined dimensions (kept from original example)
batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2


def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda")]


def get_init_inputs():
    return [in_features, out_features, dropout_p]