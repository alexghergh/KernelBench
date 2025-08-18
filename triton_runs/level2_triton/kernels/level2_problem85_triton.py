import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------- Triton kernels --------------------------- #

@triton.jit
def scale_kernel(
    x_ptr,          # *f32, input  (N*C*H*W)
    scale_ptr,      # *f32, scale (C)
    out_ptr,        # *f32, output
    n_elements: tl.int32,
    hw: tl.int32,   # H * W
    channels: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = offsets
    c_idx = ((idx // hw) % channels)

    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    s = tl.load(scale_ptr + c_idx, mask=mask, other=1.0)

    out = x * s
    tl.store(out_ptr + idx, out, mask=mask)


@triton.jit
def clamp_kernel(
    x_ptr,          # *f32
    out_ptr,        # *f32
    n_elements: tl.int32,
    clamp_min: tl.float32,
    clamp_max: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, clamp_min)
    x = tl.minimum(x, clamp_max)
    tl.store(out_ptr + offsets, x, mask=mask)


# --------------------------- Python wrappers -------------------------- #

def triton_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and scale.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    scale = scale.contiguous().view(-1)

    N, C, H, W = x.shape
    hw = H * W
    n_elements = x.numel()

    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    scale_kernel[grid](
        x, scale, out,
        n_elements,
        hw,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_clamp(x: torch.Tensor, clamp_min: float, clamp_max: float) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()

    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    clamp_kernel[grid](
        x, out,
        n_elements,
        clamp_min,
        clamp_max,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# --------------------------- Optimized model -------------------------- #

class ModelNew(nn.Module):
    """
    Optimized model that leverages custom Triton kernels for per-channel scaling
    and final clamping. Convolution, GroupNorm, and MaxPool2d are kept as
    cuDNN-optimized PyTorch modules.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        scale_shape,
        maxpool_kernel_size,
        clamp_min,
        clamp_max,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = triton_scale(x, self.scale)
        x = self.maxpool(x)
        x = triton_clamp(x, self.clamp_min, self.clamp_max)
        return x