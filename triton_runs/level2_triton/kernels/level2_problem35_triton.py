import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ------------------------------------------
# Triton kernels
# ------------------------------------------

@triton.jit
def hardswish_sub_kernel(
    x_ptr,                # *pointer* to input
    out_ptr,              # *pointer* to output
    subtract_value,       # scalar to subtract
    n_elements,           # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x = x - subtract_value
    relu6 = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
    y = x * relu6 / 6.0

    tl.store(out_ptr + offsets, y, mask=mask)


def triton_hardswish_sub(x: torch.Tensor, subtract_value: float):
    """
    Subtract `subtract_value` from `x` and apply HardSwish in a single Triton pass.
    Falls back to torch implementation on CPU.
    """
    if not x.is_cuda:
        return F.hardswish(x - subtract_value)

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    hardswish_sub_kernel[grid](
        x_contig,
        out,
        subtract_value,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def mish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x))
    # tanh(sp) = 2*sigmoid(2*sp) - 1
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    y = x * tanh_sp

    tl.store(out_ptr + offsets, y, mask=mask)


def triton_mish(x: torch.Tensor):
    """Applies Mish activation via Triton. Falls back to torch on CPU."""
    if not x.is_cuda:
        return F.mish(x)

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    mish_kernel[grid](
        x_contig,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ------------------------------------------
# Optimized model
# ------------------------------------------

class ModelNew(nn.Module):
    """
    Optimized version of the original model using custom Triton kernels
    for (x - subtract_value) + HardSwish and Mish activations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        # Ensure layers are on the correct device
        if self.conv.weight.device != x.device:
            self.conv = self.conv.to(x.device)
            self.pool = self.pool.to(x.device)

        x = self.conv(x)
        x = triton_hardswish_sub(x, self.subtract_value)
        x = self.pool(x)
        x = triton_mish(x)
        return x


# ------------------------------------------
# Helper functions (same signatures as original)
# ------------------------------------------

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]