import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------- #
#                               Triton Kernels                                 #
# ---------------------------------------------------------------------------- #

@triton.jit
def divide_const_kernel(
    x_ptr,           # *const T
    out_ptr,         # *T
    n_elements,      # int32
    divisor,         # float32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x / divisor
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_divide_const(x: torch.Tensor, divisor: float):
    """
    Element-wise division by a constant, executed with Triton.
    """
    assert x.is_cuda, "Input must be on CUDA device."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1 << 10  # 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    divide_const_kernel[grid](x, out, n_elements, divisor, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def add_bias_sum_kernel(
    x_ptr,            # *const T   (N, C) flattened input
    bias_ptr,         # *const T   (C,)
    out_ptr,          # *T         (N,)
    stride_n,         # int32      C
    C,                # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # Each program handles one batch element (N)
    offs_c = tl.arange(0, BLOCK_SIZE)
    mask = offs_c < C

    x = tl.load(x_ptr + pid * stride_n + offs_c, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs_c, mask=mask, other=0.0)

    partial = x + b
    acc = tl.sum(partial, axis=0)

    tl.store(out_ptr + pid, acc)


def triton_add_bias_sum(x: torch.Tensor, bias: torch.Tensor):
    """
    Fuses bias addition and channel-wise summation.

    Args:
        x   : Tensor of shape (N, C, 1, 1, 1) after global average pooling
        bias: Tensor of shape (C, 1, 1, 1)

    Returns:
        Tensor of shape (N, 1, 1, 1)
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must reside on CUDA device."
    N, C = x.shape[0], x.shape[1]

    # Flatten tensors so that each row is a channel vector
    x_flat = x.view(N, C).contiguous()
    bias_flat = bias.view(C).contiguous()

    out = torch.empty(N, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 32  # Should cover typical C; change if C > 32
    grid = lambda meta: (N,)

    add_bias_sum_kernel[grid](
        x_flat,
        bias_flat,
        out,
        x_flat.stride(0),
        C,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.view(N, 1, 1, 1)


# ---------------------------------------------------------------------------- #
#                             Optimized Model                                  #
# ---------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    """
    Optimized version of the original model that replaces the element-wise
    division and the bias-add + channel-sum sequence with custom Triton kernels.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        divisor,
        pool_size,
        bias_shape,
        sum_dim,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim  # kept for API parity

    def forward(self, x):
        x = self.conv(x)
        x = triton_divide_const(x, self.divisor)
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = triton_add_bias_sum(x, self.bias)
        return x


# ---------------------------------------------------------------------------- #
#                         Helpers for Benchmarking                             #
# ---------------------------------------------------------------------------- #

batch_size = 128
in_channels = 8
out_channels = 16
depth = 16
height = width = 64
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        divisor,
        pool_size,
        bias_shape,
        sum_dim,
    ]