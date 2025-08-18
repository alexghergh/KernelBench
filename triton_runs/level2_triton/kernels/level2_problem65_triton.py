import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sigmoid_sum_kernel(
    x_ptr,                # pointer to input tensor
    out_ptr,              # pointer to output tensor (per-batch results)
    n_elements,           # total number of elements in x
    elems_per_batch,      # number of elements that belong to one batch sample
    BLOCK_SIZE: tl.constexpr,
):
    # Global program id – each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)          # vector of positions
    mask = offsets < n_elements                               # bounds check

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)        # load
    y = 1.0 / (1.0 + tl.exp(-x))                              # sigmoid

    batch_idx = offsets // elems_per_batch                    # figure out which batch each element belongs to
    tl.atomic_add(out_ptr + batch_idx, y, mask=mask)          # accumulate results per batch sample


def triton_sigmoid_sum(x: torch.Tensor):
    """
    Fuses sigmoid activation with the final reduction (sum over C, H, W) into one Triton kernel.
    Returns a tensor of shape (batch_size,)
    """
    assert x.is_cuda, "Input must reside on CUDA device."
    x = x.contiguous()

    batch_size = x.shape[0]
    elems_per_batch = x[0].numel()
    n_elements = x.numel()

    out = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    sigmoid_sum_kernel[grid](
        x, out,
        n_elements, elems_per_batch,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      1. Uses PyTorch's highly-tuned Conv2d and AvgPool2d operators.
      2. Replaces the sigmoid + sum with a single custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = triton_sigmoid_sum(x)
        return x


# -------------------------------------------------------------------------
# Helpers required by the evaluation harness (unchanged from the baseline)
# -------------------------------------------------------------------------
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]