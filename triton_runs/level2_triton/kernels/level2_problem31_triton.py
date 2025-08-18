import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr,               # Pointer to Conv2d output
    bias_ptr,            # Pointer to bias tensor (C,)
    out_ptr,             # Pointer to output tensor
    constant_value,      # Scalar for torch.min
    scaling_factor,      # Scalar scaling factor
    HW,                  # H * W (spatial size)
    C,                   # Number of channels
    BLOCK_SIZE: tl.constexpr,  # Tile size
):
    # Program IDs correspond to (batch, channel, tile)
    b = tl.program_id(0)
    c = tl.program_id(1)
    tile_id = tl.program_id(2)

    # Offsets within the spatial tile
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < HW

    # Compute the flat indices into the NCHW tensor
    idxs = ((b * C + c) * HW) + offs

    # Load input activations
    x = tl.load(x_ptr + idxs, mask=mask, other=0.0)

    # Fused operations: clamp, bias add, scale
    x = tl.minimum(x, constant_value)
    bias_val = tl.load(bias_ptr + c)
    x = (x + bias_val) * scaling_factor

    # Store results
    tl.store(out_ptr + idxs, x, mask=mask)


def fused_postprocess(
    x: torch.Tensor,
    bias: torch.Tensor,
    constant_value: float,
    scaling_factor: float,
):
    """
    Applies torch.min, bias add, and scaling in a single Triton kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    x = x.contiguous()
    bias = bias.contiguous().view(-1)

    B, C, H, W = x.shape
    HW = H * W
    BLOCK_SIZE = 1024
    HW_TILES = (HW + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    grid = (B, C, HW_TILES)
    fused_kernel[grid](
        x, bias, out,
        constant_value, scaling_factor,
        HW, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses post-convolution ops (min, bias add, scale)
    into a single custom Triton kernel.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        constant_value,
        bias_shape,
        scaling_factor,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = float(constant_value)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = fused_postprocess(x, self.bias, self.constant_value, self.scaling_factor)
        return x


# ---------------------------------------------------------------------------
# Helpers for benchmarking / initialization (kept same signature as original)
# ---------------------------------------------------------------------------
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        constant_value,
        bias_shape,
        scaling_factor,
    ]