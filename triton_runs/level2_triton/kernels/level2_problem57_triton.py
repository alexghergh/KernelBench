import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_relu_hardswish_kernel(
    x_ptr,          # pointer to input tensor
    out_ptr,        # pointer to output tensor
    n_elements,     # number of elements to process
    BLOCK_SIZE: tl.constexpr,  # meta-parameter: how many elements each block handles
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # ReLU
    relu = tl.maximum(x, 0.0)

    # Hard-Swish scaling factor: clamp((x + 3) / 6, 0, 1)
    scale = (relu + 3.0) / 6.0
    scale = tl.minimum(tl.maximum(scale, 0.0), 1.0)

    out = relu * scale
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_relu_hardswish(x: torch.Tensor) -> torch.Tensor:
    """
    Fused ReLU + HardSwish activation.
    Falls back to the PyTorch implementation on CPU.
    """
    if not x.is_cuda:
        y = torch.relu(x)
        return y * torch.clamp((y + 3) / 6, 0, 1)

    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_relu_hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      • Keeps the convolution in PyTorch.
      • Replaces ReLU + HardSwish with a single Triton fused kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = fused_relu_hardswish(x)
        return x


# Default parameters / shapes
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]