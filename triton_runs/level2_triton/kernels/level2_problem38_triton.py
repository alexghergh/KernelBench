import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------- Triton Kernels ---------------------------- #
@triton.jit
def clamp_kernel(
    x_ptr,              # *f32
    out_ptr,            # *f32
    min_val: tl.constexpr,   # f32 scalar
    max_val: tl.constexpr,   # f32 scalar
    n_elements,         # i32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.minimum(tl.maximum(x, min_val), max_val)
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_clamp(x: torch.Tensor, clamp_min: float, clamp_max: float):
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elems = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elems + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    clamp_kernel[grid](x, out, clamp_min, clamp_max, n_elems, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def scale_kernel(
    x_ptr,              # *f32
    scale_ptr,          # *f32, length = C
    out_ptr,            # *f32
    n_elements,         # i32
    inner_size,         # i32 = D*H*W
    C,                  # i32 = channels
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # channel index = ((index // inner_size) % C)
    ch_idx = (offsets // inner_size) % C
    scale_val = tl.load(scale_ptr + ch_idx, mask=mask)

    out = x * scale_val
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scale(x: torch.Tensor, scale: torch.Tensor):
    assert x.is_cuda and scale.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    scale_vec = scale.contiguous().view(-1)  # shape (C,)
    C = scale_vec.shape[0]
    n_elems = x.numel()
    inner_size = x.shape[-3] * x.shape[-2] * x.shape[-1]  # D*H*W
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elems + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    scale_kernel[grid](x, scale_vec, out, n_elems, inner_size, C, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ------------------------------ Model ----------------------------------- #
class ModelNew(nn.Module):
    """
    Optimized model that keeps heavyweight ConvTranspose3d in PyTorch but
    replaces clamping and channel-wise scaling with custom Triton kernels.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        clamp_min,
        clamp_max
    ):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.register_parameter("scale", nn.Parameter(torch.ones(1, out_channels, 1, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = triton_clamp(x, self.clamp_min, self.clamp_max)

        b, c, d, h, w = x.shape
        x = x.view(b, c, -1)
        x = torch.softmax(x, dim=2)
        x = x.view(b, c, d, h, w)

        x = triton_scale(x, self.scale)
        return x


# ---------------------- Helpers for Bench Framework --------------------- #
batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 32, 64, 64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        clamp_min,
        clamp_max,
    ]