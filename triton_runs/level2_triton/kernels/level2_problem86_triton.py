import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_div_gelu_kernel(
    A_ptr,  # (M, K) input
    B_ptr,  # (K, N) weight (transposed)
    C_ptr,  # (M, N) output
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bias_ptr,            # (N,)
    divisor,             # scalar float
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (offs_k[None, :] + k_offset) * stride_ak)
        b_ptrs = B_ptr + ((offs_k[:, None] + k_offset) * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] + k_offset < K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] + k_offset < K) & (offs_n[None, :] < N),
                    other=0.0)

        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Divide by scalar
    acc = acc / divisor

    # GELU activation
    sqrt_half = 0.7071067811865475  # 1/sqrt(2)
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * sqrt_half))

    # Store the result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_fused_linear_div_gelu(x: torch.Tensor,
                                 weight: torch.Tensor,
                                 bias: torch.Tensor,
                                 divisor: float):
    """
    Fused Linear (matmul + bias) + division + GELU implemented with Triton.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    weight_t = weight.t().contiguous()  # (K, N)
    bias = bias.contiguous()

    M, K = x.shape
    N = weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))

    fused_linear_div_gelu_kernel[grid](
        x, weight_t, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
        bias,
        divisor,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with a fused Triton kernel performing:
    Linear (matmul + bias) + division by scalar + GELU activation.
    """
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.divisor = divisor

        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return triton_fused_linear_div_gelu(x, self.weight, self.bias, self.divisor)


# ----------------------------------------------------------------------
# Functions to match the original interface
batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size, device='cuda')]


def get_init_inputs():
    return [input_size, output_size, divisor]