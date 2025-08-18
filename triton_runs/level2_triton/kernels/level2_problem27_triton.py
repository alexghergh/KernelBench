import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardswish_kernel(
    x_ptr,       # input
    out_ptr,     # output
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tmp = x + 3.0
    tmp = tl.minimum(tl.maximum(tmp, 0.0), 6.0)
    y = x * tmp / 6.0

    tl.store(out_ptr + offsets, y, mask=mask)


def triton_hardswish(x: torch.Tensor, block_size: int = 1024):
    """HardSwish implemented with a Triton kernel."""
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
        1. Conv3D
        2. HardSwish (Triton)
        3. GroupNorm
        4. Mean over spatial dims
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)                       # (B, C, D, H, W)
        x = triton_hardswish(x)                # Triton HardSwish
        x = self.group_norm(x)                 # GroupNorm
        x = torch.mean(x, dim=[2, 3, 4])       # Spatial mean → (B, C)
        return x


# === helpers for benchmarking harness ===
batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device="cuda")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]