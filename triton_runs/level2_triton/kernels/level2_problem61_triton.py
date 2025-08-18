import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,        # Pointer to input
    out_ptr,      # Pointer to output
    n_elements,   # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d + Triton ReLU + GroupNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_relu(x)
        x = self.group_norm(x)
        return x


# ---- Helpers for benchmarking ------------------------------------------------
batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W, device="cuda")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]