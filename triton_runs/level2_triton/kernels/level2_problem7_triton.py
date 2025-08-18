import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_act_bias_kernel(
    x_ptr,       # *f32
    bias_ptr,    # *f32
    out_ptr,     # *f32
    C,           # int32
    DHW,         # int32 (depth * height * width)
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)          # index over (N * C)
    pid_blk = tl.program_id(1)        # block inside the (D*H*W) dimension

    offsets = pid_blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < DHW

    flat_idx = pid_c * DHW + offsets

    # Load input
    x = tl.load(x_ptr + flat_idx, mask=mask, other=0.0)

    # ReLU (LeakyReLU(0.01) after ReLU is equivalent to ReLU)
    x = tl.maximum(x, 0.0)

    # GELU (exact, erf–based)
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    # Sigmoid
    x = 1.0 / (1.0 + tl.exp(-x))

    # Add channel bias (broadcasted)
    c = pid_c % C
    b = tl.load(bias_ptr + c)
    x = x + b

    # Store result
    tl.store(out_ptr + flat_idx, x, mask=mask)


def fused_act_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Apply ReLU -> GELU -> Sigmoid -> +bias in a single Triton kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    x = x.contiguous()
    bias_1d = bias.flatten().contiguous()  # (C,)

    N, C, D, H, W = x.shape
    DHW = D * H * W
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = lambda meta: (N * C, triton.cdiv(DHW, meta['BLOCK_SIZE']))

    fused_act_bias_kernel[grid](
        x, bias_1d, out,
        C, DHW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel to fuse activation stack
    and bias addition following a 3-D convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_act_bias(x, self.bias)
        return x


# --- Helpers expected by the harness -------------------------------------------------
batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device='cuda')]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]