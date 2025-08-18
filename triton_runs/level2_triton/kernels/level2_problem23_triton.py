import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_kernel(
    x_ptr,           # Pointer to input tensor
    out_ptr,         # Pointer to per–batch partial sums
    total_elems,     # Total number of elements in `x`
    elems_per_batch, # Number of elements in one batch sample
    BLOCK_SIZE: tl.constexpr,
):
    # Global element index handled by this program instance + lanes
    start = tl.program_id(0) * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elems

    # Load data
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Figure out which batch sample every element belongs to
    batch_ids = offsets // elems_per_batch

    # Accumulate into the output vector using atomic adds
    tl.atomic_add(out_ptr + batch_ids, vals, mask=mask)


def triton_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean over all dimensions except the batch dimension with Triton.
    Returns a tensor of shape (batch_size,).
    """
    assert x.is_cuda, "Input tensor must reside on CUDA"
    x = x.contiguous()

    B = x.shape[0]
    elems_per_batch = x[0].numel()
    total_elems = x.numel()

    # Buffer for the per-batch sums
    sums = torch.zeros(B, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 1024
    grid = lambda meta: ((total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _reduce_sum_kernel[grid](
        x, sums,
        total_elems, elems_per_batch,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Convert sums to means
    sums /= elems_per_batch
    return sums


class ModelNew(nn.Module):
    """
    Optimized variant of the reference model:
      1. Keeps Conv3d and GroupNorm as-is.
      2. Replaces the final reduction (mean over C, D, H, W) with a custom
         Triton kernel that parallelizes the reduction across the whole tensor
         and uses atomic adds for accumulation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        return triton_mean(x)


# -------------------------------------------------------------------------
# Helper functions mimicking the originals so the outside interface matches
# -------------------------------------------------------------------------
batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W, device='cuda')]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]