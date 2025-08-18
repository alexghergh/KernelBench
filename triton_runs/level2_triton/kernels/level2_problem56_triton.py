import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sigmoid_row_sum_kernel(
    x_ptr,          # * pointer to the flattened input matrix (B * H elements)
    out_ptr,        # * pointer to the output vector (B elements)
    n_cols,         #   number of columns (H)
    total_elems,    #   total elements in the flattened matrix (B * H)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elems

    # Load a block of data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply sigmoid
    x = 1.0 / (1.0 + tl.exp(-x))

    # Compute row indices for each element
    row_ids = offsets // n_cols

    # Atomically accumulate the sigmoid values into the per-row sums
    tl.atomic_add(out_ptr + row_ids, x, mask=mask)


def triton_sigmoid_row_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Fuses sigmoid activation and row-wise summation using a Triton kernel.

    Args:
        x: Tensor on CUDA of shape (batch_size, hidden_size)

    Returns:
        Tensor of shape (batch_size, 1) where each entry is
        sum(sigmoid(x_i), dim=1)
    """
    assert x.is_cuda, "Input must reside on CUDA"
    x = x.contiguous()
    batch_size, n_cols = x.shape
    total_elems = x.numel()

    # Flatten for 1-D indexing inside the kernel
    x_flat = x.view(-1)

    # Output buffer for the row-wise sums
    out = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: ((total_elems + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    sigmoid_row_sum_kernel[grid](
        x_flat,
        out,
        n_cols,
        total_elems,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.unsqueeze(1)


class ModelNew(nn.Module):
    """
    Optimized model: keeps the Linear layer, but replaces the
    sigmoid + sum with a fused Triton kernel.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = self.linear(x)
        x = triton_sigmoid_row_sum(x)
        return x


# ----------------- helpers for benchmarking -----------------
batch_size = 128
input_size = 32768
hidden_size = 32768


def get_inputs():
    return [torch.rand(batch_size, input_size, device="cuda")]


def get_init_inputs():
    return [input_size, hidden_size]