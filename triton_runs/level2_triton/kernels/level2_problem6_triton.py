import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _softmax_fwd_kernel(
    x_ptr,        # input  pointer
    out_ptr,      # output pointer
    n_cols,       # number of channels (C)
    BLOCK_SIZE: tl.constexpr,
):
    # Program id == row id (all dimensions except channels)
    row_id = tl.program_id(0)

    # Column indices processed by this program
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load data
    ptrs = x_ptr + row_id * n_cols + col_offsets
    x = tl.load(ptrs, mask=mask, other=-float('inf'))

    # Row-wise softmax
    row_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - row_max)
    row_sum = tl.sum(x_exp, axis=0)
    out = x_exp / row_sum

    # Store result
    tl.store(out_ptr + row_id * n_cols + col_offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Channel-wise softmax (dim=1) for 5-D NCDHW tensors using a Triton kernel.
    """
    assert x.is_cuda, "Input tensor must reside on GPU"
    assert dim == 1, "triton_softmax currently supports softmax over dim=1 only"
    assert x.dim() == 5, "Expected a 5-D tensor (N, C, D, H, W)"

    # Move channels to the last dimension to make them contiguous
    x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
    n_rows = x_perm.numel() // x_perm.shape[-1]
    n_cols = x_perm.shape[-1]

    # Prepare output
    out_perm = torch.empty_like(x_perm)

    # Choose a power-of-two block size up to 1024
    BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    grid = lambda meta: (n_rows,)

    _softmax_fwd_kernel[grid](
        x_perm, out_perm,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Restore original layout
    return out_perm.permute(0, 4, 1, 2, 3)


class ModelNew(nn.Module):
    """
    Optimized model that replaces the channel-wise Softmax with a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = triton_softmax(x, dim=1)
        x = self.pool1(x)
        x = self.pool2(x)
        return x


# ----------------------------------------------------------------------------- #
# Helpers (kept identical to the original interface)
# ----------------------------------------------------------------------------- #
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device='cuda')]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]