import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr,            # Pointer to input tensor
    bias_ptr,         # Pointer to bias tensor (flattened)
    out_ptr,          # Pointer to output tensor
    n_elems,          # Total number of elements in x/out
    hw,               # Height * Width of the feature-map
    C,                # Number of channels
    scaling_factor,   # Scaling factor (float32)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elems

    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Figure out the channel index for every element
    tmp = offs // hw          # tmp = n*C + c
    c_idx = tmp % C           # isolate channel index
    bias_val = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # Element-wise computations (fused)
    y = x + bias_val
    y = tl.where(y < 0.0, 0.0, y)   # clamp min
    y = tl.where(y > 1.0, 1.0, y)   # clamp max
    y = y * scaling_factor
    y = tl.where(y > 1.0, 1.0, y)   # second clamp
    y = y / scaling_factor

    # Store the result
    tl.store(out_ptr + offs, y, mask=mask)


def fused_bias_clamp_scale(x: torch.Tensor,
                           bias: torch.Tensor,
                           scaling_factor: float) -> torch.Tensor:
    """
    Fuses: +bias -> clamp -> *sf -> clamp -> /sf
    into one Triton kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."

    x = x.contiguous()
    bias_flat = bias.contiguous().view(-1)  # (C,1,1) → (C,)

    out = torch.empty_like(x)

    n_elems = x.numel()
    hw = x.shape[-1] * x.shape[-2]   # W * H
    C = x.shape[1]

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elems + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_kernel[grid](
        x, bias_flat, out,
        n_elems, hw, C,
        scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the original model:
    keeps the ConvTranspose2d layer in PyTorch, but fuses the subsequent
    element-wise operations into a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_bias_clamp_scale(x, self.bias, self.scaling_factor)
        return x


# ---------- Helpers (unchanged) ----------
batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding,
            output_padding, bias_shape, scaling_factor]