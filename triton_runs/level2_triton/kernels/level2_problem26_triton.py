import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_add_hardswish_kernel(
    x_ptr,          # Pointer to tensor from ConvTranspose3d
    add_ptr,        # Pointer to tensor to add
    out_ptr,        # Pointer to output tensor
    n_elements,     # Number of elements in tensors
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each Triton program processes BLOCK_SIZE elements and applies:
        y = x + add
        hswish = y * relu6(y + 3) / 6
        out = y * hswish
    which fuses the add and the HardSwish–based computation.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    add = tl.load(add_ptr + offsets, mask=mask)

    y = x + add
    relu6 = tl.minimum(tl.maximum(y + 3.0, 0.0), 6.0)
    hswish = y * relu6 / 6.0
    out = y * hswish

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_add_hardswish(x: torch.Tensor, add: torch.Tensor) -> torch.Tensor:
    """
    Wraps the Triton kernel launch that fuses:
        1. Addition of two tensors
        2. y * HardSwish(y)
    """
    assert x.is_cuda and add.is_cuda, "Inputs must be CUDA tensors."
    assert x.shape == add.shape, "Input shapes must match."

    x = x.contiguous()
    add = add.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_add_hardswish_kernel[grid](x, add, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps the PyTorch ConvTranspose3d layer but
    replaces the subsequent add + HardSwish computation with a single
    fused Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        # Extra bias parameter kept to mirror original architecture (unused in forward)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        x = triton_fused_add_hardswish(x, add_input)
        return x


# -----------------------------------------------------------------------------
# Helpers to replicate the original interface
# -----------------------------------------------------------------------------
batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)


def get_inputs():
    return [
        torch.rand(batch_size, in_channels, D, H, W, device="cuda"),
        torch.rand(batch_size, out_channels, D * stride, H * stride, W * stride, device="cuda"),
    ]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]