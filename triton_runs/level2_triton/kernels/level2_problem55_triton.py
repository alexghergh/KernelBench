import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2_sum_scale_kernel(
    x_ptr,          # *float32 [B, N]
    out_ptr,        # *float32 [B]
    scale,          # float32
    N,              # int32  = number of features
    N_PAIRS,        # int32  = N // 2
    BLOCK_SIZE: tl.constexpr,  # number of pairs processed by each program
):
    batch_id = tl.program_id(0)
    pair_block = tl.program_id(1)

    # Compute pair indices handled by this program
    pair_start = pair_block * BLOCK_SIZE
    pair_offsets = pair_start + tl.arange(0, BLOCK_SIZE)
    mask = pair_offsets < N_PAIRS

    # Map pair index -> element indices in original feature dimension
    idx1 = pair_offsets * 2
    idx2 = idx1 + 1

    # Base pointer for the current row
    row_ptr = x_ptr + batch_id * N

    # Load the two elements of each pair
    a = tl.load(row_ptr + idx1, mask=mask, other=0.0)
    b = tl.load(row_ptr + idx2, mask=mask, other=0.0)

    # Max-pool over each pair
    m = tl.maximum(a, b)

    # Sum the maxima for this block
    partial_sum = tl.sum(m, axis=0)

    # Atomically accumulate the scaled partial sum into the output
    tl.atomic_add(out_ptr + batch_id, partial_sum * scale)


def triton_maxpool2_sum_scale(x: torch.Tensor, scale: float, block_size: int = 1024):
    """
    Fused Triton implementation of:
        x = max_pool1d(x.unsqueeze(1), kernel_size=2).squeeze(1)
        x = torch.sum(x, dim=1)
        x = x * scale
    Args:
        x    : (batch, features) CUDA tensor
        scale: scalar multiplier
    Returns:
        (batch,) CUDA tensor
    """
    assert x.is_cuda, "Input must reside on CUDA."
    assert x.dim() == 2, "Input must be 2-D (batch, features)."
    B, N = x.shape
    assert N % 2 == 0, "Feature dimension must be even for kernel_size=2 pooling."

    n_pairs = N // 2
    out = torch.zeros(B, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        B,
        (n_pairs + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    maxpool2_sum_scale_kernel[grid](
        x,
        out,
        scale,
        N,
        n_pairs,
        BLOCK_SIZE=block_size,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps the high-performance cuBLAS-backed Linear layer
    but replaces MaxPool1d + sum + scaling with a single fused Triton kernel.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        assert kernel_size == 2, "Fused Triton kernel supports kernel_size=2 only."
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        x = self.matmul(x)
        x = triton_maxpool2_sum_scale(x, self.scale_factor)
        return x


# ----------------------------------------------------------------------
# Helpers (kept identical to the original interface for seamless usage)
# ----------------------------------------------------------------------
batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5


def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda")]


def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]