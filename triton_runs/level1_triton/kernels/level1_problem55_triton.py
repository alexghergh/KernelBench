import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


#########################
#  Triton MatMul Kernel #
#########################
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,                       # pointers to matrices
    M, N, K,                                   # dimensions
    stride_am, stride_ak,                      # strides for A
    stride_bk, stride_bn,                      # strides for B
    stride_cm, stride_cn,                      # strides for C
    BLOCK_M: tl.constexpr,                     # tile sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
        b_ptrs = B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication wrapper: (M, K) x (K, N) -> (M, N)
    Both inputs must be float32 CUDA tensors.
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Only float32 supported"
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Incompatible matmul shapes"

    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


#########################
#  Triton Conv2d Layer  #
#########################
class Conv2dTriton(nn.Module):
    """
    A 2-D convolution implemented via `unfold` + Triton GEMM.
    Supports: stride, padding. (dilation > 1 or groups > 1 not yet supported)
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
        assert groups == 1, "Grouped convolution not supported"
        assert dilation == 1, "Dilation not supported"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * (self.kernel_size ** 2)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        returns: (B, C_out, H_out, W_out)
        """
        B, _, H, W = x.shape
        K = self.kernel_size

        # Im2col
        patches = F.unfold(
            x,
            kernel_size=K,
            padding=self.padding,
            stride=self.stride,
        )  # (B, C_in*K*K, L) where L = H_out*W_out

        weight_2d = self.weight.view(self.out_channels, -1).contiguous()  # (C_out, C_in*K*K)

        outputs = []
        for b in range(B):
            cols = patches[b].contiguous()  # (C_in*K*K, L)
            out_b = triton_matmul(weight_2d, cols)  # (C_out, L)
            if self.bias is not None:
                out_b += self.bias[:, None]
            outputs.append(out_b)

        out = torch.stack(outputs, dim=0)  # (B, C_out, L)

        # Reshape to (B, C_out, H_out, W_out)
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1
        out = out.view(B, self.out_channels, H_out, W_out)
        return out


#########################
#      ModelNew         #
#########################
class ModelNew(nn.Module):
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
        self.conv2d = Conv2dTriton(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)