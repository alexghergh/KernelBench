import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_leaky_relu_mul_kernel(
    x_ptr,       # *float32
    out_ptr,     # *float32
    mul_ptr,     # *float32
    S,           # spatial size (D*H*W)
    C,           # number of channels
    BLOCK_SIZE: tl.constexpr,  # number of elements processed by each program
):
    # Program identifiers
    row_id = tl.program_id(0)          # index over (N * C)
    block_id = tl.program_id(1)        # index over tiles of size BLOCK_SIZE within a row

    # Channel index for this row and its multiplier value
    c = row_id % C
    m = tl.load(mul_ptr + c)           # scalar

    # Compute element offsets this program will process
    offs_in_row = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_in_row < S             # stay within spatial bounds

    # Global (flattened) offsets
    offs = row_id * S + offs_in_row

    # Load, apply first LeakyReLU, multiply, apply second LeakyReLU, store
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    x = tl.where(x > 0.0, x, 0.2 * x)  # first LeakyReLU
    x = x * m                          # channel-wise multiply
    x = tl.where(x > 0.0, x, 0.2 * x)  # second LeakyReLU

    tl.store(out_ptr + offs, x, mask=mask)


def fused_leaky_relu_mul(x: torch.Tensor, multiplier: torch.Tensor) -> torch.Tensor:
    """
    Fuses: LeakyReLU → channel-wise multiplication → LeakyReLU
    into a single Triton kernel.

    Args:
        x:           Tensor of shape [N, C, D, H, W] (CUDA, contiguous)
        multiplier:  Tensor of shape [C, 1, 1, 1]  (CUDA)

    Returns:
        Tensor with the same shape as `x`.
    """
    assert x.is_cuda and multiplier.is_cuda, "Tensors must be on CUDA."

    x = x.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W                        # spatial element count per channel
    BLOCK_SIZE = 128                     # Tunable

    # Flattened views for kernel launch
    x_flat = x.view(-1)
    out = torch.empty_like(x)
    out_flat = out.view(-1)
    mul_flat = multiplier.view(-1).contiguous()

    # Grid: one program per (N * C) row and per spatial tile of size BLOCK_SIZE
    grid = (N * C, triton.cdiv(S, BLOCK_SIZE))

    fused_leaky_relu_mul_kernel[grid](
        x_flat, out_flat, mul_flat,
        S, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the original model.
    ConvTranspose3d → fused(LeakyReLU * param * LeakyReLU) → MaxPool3d
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, multiplier_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Learnable per-channel multiplier
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_leaky_relu_mul(x, self.multiplier)
        x = self.max_pool(x)
        return x


# Utility functions (unchanged)

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size,
        stride, padding, output_padding, multiplier_shape
    ]