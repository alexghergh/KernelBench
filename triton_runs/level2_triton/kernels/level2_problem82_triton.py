import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tanh_scale_bias_kernel(
    x_ptr,            # Input tensor
    bias_ptr,         # Bias tensor (C, 1, 1)
    out_ptr,          # Output tensor
    n_elements,       # Total elements in x
    hw,               # H * W (spatial area)
    C,                # Channels
    scaling,          # Scaling factor
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # tanh(x) implemented via exponentials to avoid relying on libdevice tanh
    exp_pos = tl.exp(x)
    exp_neg = tl.exp(-x)
    t = (exp_pos - exp_neg) / (exp_pos + exp_neg)  # tanh(x)

    # Channel index for bias broadcasting
    c_idx = (offsets // hw) % C
    bias_val = tl.load(bias_ptr + c_idx, mask=mask)

    out = t * scaling + bias_val
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_tanh_scale_bias(x: torch.Tensor, scaling: float, bias: torch.Tensor) -> torch.Tensor:
    """
    Fuses tanh activation, scaling, and bias addition using a Triton kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Input and bias must be on CUDA"
    x = x.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)

    N, C, H, W = x.shape
    hw = H * W
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    tanh_scale_bias_kernel[grid](
        x,
        bias,
        out,
        n_elements,
        hw,
        C,
        scaling,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused tanh + scaling + bias addition implemented in Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = fused_tanh_scale_bias(x, self.scaling_factor, self.bias)
        x = self.max_pool(x)
        return x


# ------------------------------------------------------------------------
# Helpers (unchanged signatures)
# ------------------------------------------------------------------------
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]