import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ------------------------- Triton MatMul Kernel ------------------------- #
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)    # block index for rows of C
    pid_n = tl.program_id(1)    # block index for cols of C

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak)
        b_ptrs = B_ptr + ((k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A @ B using a custom Triton kernel.
    A: (M, K) float32 contiguous
    B: (K, N) float32 contiguous
    Returns C: (M, N) float32
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only fp32 supported"

    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Incompatible matmul dimensions"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    _matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return C


# --------------------------- Triton Conv1d --------------------------- #
def _triton_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    """
    Simplified 1-D convolution implemented via im2col + Triton matmul.
    Supports:
      • stride = 1
      • dilation = 1
      • groups = 1
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
    assert stride == 1, "Only stride = 1 supported in custom kernel"
    assert dilation == 1, "Only dilation = 1 supported in custom kernel"

    B, C_in, L_in = x.shape
    C_out, C_in_w, K = weight.shape
    assert C_in == C_in_w, "Incompatible channels"

    # Padding
    if padding:
        x = F.pad(x, (padding, padding))

    # Output length
    L_out = x.shape[-1] - K + 1

    # im2col: (B, C_in, L_out, K)
    x_cols = (
        x.unfold(dimension=2, size=K, step=1)
        .contiguous()
        .permute(0, 2, 1, 3)
        .reshape(B * L_out, C_in * K)
        .contiguous()
    )

    # Prepare weight: (K*C_in, C_out)
    w_flat = weight.reshape(C_out, C_in * K).t().contiguous()

    # MatMul
    out = _triton_matmul(x_cols, w_flat)  # (B*L_out, C_out)

    # Bias
    if bias is not None:
        out += bias.view(1, -1)

    # Reshape back to NCHW-like: (B, C_out, L_out)
    out = out.view(B, L_out, C_out).permute(0, 2, 1).contiguous()
    return out


# ----------------------------- ModelNew ----------------------------- #
class ModelNew(nn.Module):
    """
    Optimized 1-D convolution layer using custom Triton kernels.
    Currently supports:
      • groups = 1
      • stride = 1
      • dilation = 1
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert groups == 1, "groups > 1 not supported in Triton kernel"
        assert stride == 1, "stride > 1 not supported in Triton kernel"
        assert dilation == 1, "dilation > 1 not supported in Triton kernel"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Parameters
        weight = torch.empty(out_channels, in_channels, kernel_size)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
            fan_in = in_channels * kernel_size
            bound = 1 / math.sqrt(fan_in)
            bias_param = torch.empty(out_channels)
            nn.init.uniform_(bias_param, -bound, bound)
            self.bias = nn.Parameter(bias_param)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _triton_conv1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )