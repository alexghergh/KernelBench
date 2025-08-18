import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _global_avg_pool_kernel(
    x_ptr,                      # *fp32, input tensor
    out_ptr,                    # *fp32, output tensor
    multiplier,                 # fp32 scalar
    HW: tl.constexpr,           # H * W
    BLOCK_SIZE: tl.constexpr,   # number of elements loaded per program
):
    pid = tl.program_id(0)               # 1D grid: one program per (batch, channel)
    offset = pid * HW                    # starting index of the (b, c) chunk
    acc = tl.zeros((), dtype=tl.float32) # accumulator

    # Loop over the spatial dimension in BLOCK_SIZE chunks
    for i in range(0, HW, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = (i + offs) < HW
        ptrs = x_ptr + offset + i + offs
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    # Scale, divide by number of elements to get the mean and write result
    acc = acc * multiplier / HW
    tl.store(out_ptr + pid, acc)


def triton_global_avg_pool_scaled(x: torch.Tensor, multiplier: torch.Tensor):
    """
    Scales `x` by `multiplier` and performs global average pooling over the last two dims
    using a Triton kernel. Returns a tensor of shape (B, C, 1, 1).
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    B, C, H, W = x.shape
    HW = H * W

    out = torch.empty((B * C,), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 1024  # Tunable
    grid = lambda meta: (B * C,)

    _global_avg_pool_kernel[grid](
        x, out, float(multiplier.item()), HW, BLOCK_SIZE=BLOCK_SIZE
    )

    return out.view(B, C, 1, 1)


class ModelNew(nn.Module):
    """
    Optimized version of the original model.
    - Keeps the transposed convolution as‐is (cuDNN-optimised).
    - Fuses the scalar multiplication and two redundant global average poolings
      into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Store multiplier as a buffer so it moves with .to(...)
        self.register_buffer("multiplier", torch.tensor(multiplier, dtype=torch.float32))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_global_avg_pool_scaled(x, self.multiplier)
        return x


# --------------------------------------------------------------------------
# Boiler-plate utilities expected by the runner
# --------------------------------------------------------------------------
batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        multiplier,
    ]