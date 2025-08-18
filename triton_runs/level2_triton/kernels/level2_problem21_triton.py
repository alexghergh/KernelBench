import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_bias_scale_sigmoid_kernel(
    x_ptr,            # Pointer to input tensor
    bias_ptr,         # Pointer to bias (C,)
    scale_ptr,        # Pointer to scale (C,)
    out_ptr,          # Pointer to output tensor
    N,                # Batch size
    C,                # Channel count
    HW,               # H * W (spatial size per channel)
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements processed by each program
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    elements_per_batch = C * HW
    idx_in_batch = offsets % elements_per_batch
    channel_idx = idx_in_batch // HW  # Channel index for each element

    bias = tl.load(bias_ptr + channel_idx, mask=mask)
    scale = tl.load(scale_ptr + channel_idx, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    x = (x + bias) * scale
    x = 1.0 / (1.0 + tl.exp(-x))  # Sigmoid
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_bias_scale_sigmoid(
    x: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    Applies (x + bias) * scale followed by sigmoid using a Triton kernel.
    bias and scale are expected to have shape (C, 1, 1) and broadcast along H & W.
    """
    assert x.is_cuda and bias.is_cuda and scale.is_cuda, "All tensors must be on CUDA"
    x = x.contiguous()
    bias = bias.contiguous().view(-1)
    scale = scale.contiguous().view(-1)

    n, c, h, w = x.shape
    hw = h * w
    total_elems = x.numel()

    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: ((total_elems + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_bias_scale_sigmoid_kernel[grid](
        x, bias, scale, out,
        n, c, hw, total_elems,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses bias addition, scaling, and sigmoid activation
    into a single Triton kernel.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size,
        num_groups, bias_shape, scale_shape
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = fused_bias_scale_sigmoid(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x


# --------------------------------------------------------------------
# Helper functions expected by the evaluation harness
# --------------------------------------------------------------------
batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        bias_shape,
        scale_shape,
    ]