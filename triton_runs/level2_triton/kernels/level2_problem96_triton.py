import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mul_scalar_kernel(
    x_ptr,               # *f32
    out_ptr,             # *f32
    scale,               # f32
    n_elements,          # i32
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_mul_scalar(x: torch.Tensor, scale: float):
    """
    Element-wise multiplication by a scalar using Triton.
    """
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mul_scalar_kernel[grid](x, out, scale, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def avg_clamp_kernel(
    x_ptr,                        # *f32
    out_ptr,                      # *f32
    elements_per_channel,         # i32
    clamp_min,                    # f32
    clamp_max,                    # f32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)                       # program id == unique (N, C) pair
    channel_offset = pid * elements_per_channel  # starting index for this (N, C)
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((), dtype=tl.float32)

    for i in range(0, elements_per_channel, BLOCK_SIZE):
        ptrs = x_ptr + channel_offset + i + offs
        mask = (i + offs) < elements_per_channel
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    mean = acc / elements_per_channel
    mean = tl.maximum(mean, clamp_min)
    mean = tl.minimum(mean, clamp_max)
    tl.store(out_ptr + pid, mean)


def triton_global_avg_pool_clamp(
    x: torch.Tensor, clamp_min: float = 0.0, clamp_max: float = 1.0
) -> torch.Tensor:
    """
    Global average pooling over D, H, W followed by clamp using Triton.
    Returns tensor shaped (N, C, 1, 1, 1).
    """
    assert x.is_cuda and x.dim() == 5, "Input must be 5-D CUDA tensor"
    x = x.contiguous()
    N, C, D, H, W = x.shape
    elements_per_channel = D * H * W

    out_flat = torch.empty(N * C, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024
    grid = lambda meta: (N * C,)
    avg_clamp_kernel[grid](
        x,
        out_flat,
        elements_per_channel,
        clamp_min,
        clamp_max,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_flat.view(N, C, 1, 1, 1)


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for scalar multiply and
    fused global average pooling + clamp.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        scale,
        maxpool_kernel_size,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        # store scale as buffer for easy CUDA device management
        self.register_buffer("scale_tensor", torch.tensor(float(scale), dtype=torch.float32))
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.clamp_min = 0.0
        self.clamp_max = 1.0

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_mul_scalar(x, float(self.scale_tensor))
        x = self.maxpool(x)
        x = triton_global_avg_pool_clamp(x, self.clamp_min, self.clamp_max)
        return x


# ---- helper functions expected by the harness ----
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device="cuda")]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        scale,
        maxpool_kernel_size,
    ]