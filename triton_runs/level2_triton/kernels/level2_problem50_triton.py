import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def scale_bias_kernel(
    x_ptr,        # pointer to input after avg_pool3d
    bias_ptr,     # pointer to bias * scale2, size = C
    out_ptr,      # pointer to output
    weight,       # scalar = scale1 * scale2
    spatial_size, # D * H * W (number of values per (B,C) slice)
    C,            # number of channels
    BLOCK_SIZE: tl.constexpr,
):
    # Each kernel instance works on one (batch, channel) pair
    bc_idx = tl.program_id(0)                         # which (B,C) slice
    offset = tl.program_id(1) * BLOCK_SIZE            # start element within that slice
    offsets = offset + tl.arange(0, BLOCK_SIZE)       # strides inside slice
    mask = offsets < spatial_size                     # guard against OOB

    global_offsets = bc_idx * spatial_size + offsets  # absolute positions

    # Load data
    x_val = tl.load(x_ptr + global_offsets, mask=mask, other=0.0)

    # Load corresponding bias value for this channel
    c_idx = bc_idx % C
    bias_val = tl.load(bias_ptr + c_idx)

    # Fused computation:  out = x * weight + bias_val
    out_val = x_val * weight + bias_val

    # Store results
    tl.store(out_ptr + global_offsets, out_val, mask=mask)


def triton_scale_bias(
    x: torch.Tensor,
    bias: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
):
    """
    Given x (output of avg_pool3d), compute
        out = x * (scale1*scale2) + bias*scale2
    using a Triton kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Input and bias must be on CUDA"
    x = x.contiguous()
    bias = bias.contiguous()

    B, C, D, H, W = x.shape
    spatial_size = D * H * W

    # Pre-compute combined scalars/tensors
    weight = float(scale1.item() * scale2.item())          # scalar
    bias_scaled = (bias * scale2).contiguous()             # shape (C,1,1,1) or (C)

    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = (B * C, (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE)

    scale_bias_kernel[grid](
        x, bias_scaled, out,
        weight,
        spatial_size, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the original model.
    ConvTranspose3d is kept as-is (it already dispatches to cuDNN),
    while the post-pool scaling, bias addition and final scaling are
    fused into a single custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, scale1, scale2, bias_shape):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )

        self.scale1 = nn.Parameter(torch.tensor(scale1, dtype=torch.float32))
        self.scale2 = nn.Parameter(torch.tensor(scale2, dtype=torch.float32))
        self.bias   = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))

        # AvgPool3d keeps default stride=kernel_size (2) behaviour
        self.avg_pool = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)        # de-convolution
        x = self.avg_pool(x)              # average pooling (stride 2)
        x = triton_scale_bias(            # fused element-wise ops
            x, self.bias, self.scale1, self.scale2
        )
        return x