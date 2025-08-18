import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,              # Pointer to input tensor
    out_ptr,            # Pointer to output tensor
    n_elements,         # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements processed by each kernel instance
):
    # Program ID corresponds to the block index
    pid = tl.program_id(0)
    # Compute the start index of the block operated by this program
    block_start = pid * BLOCK_SIZE
    # Offsets within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to guard against out-of-bounds access
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # ReLU
    y = tl.maximum(x, 0.0)
    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    """Apply ReLU using a custom Triton kernel."""
    assert x.is_cuda, "Input tensor must reside on a CUDA device."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 2048  # Tunable; larger reduces grid size

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that replaces torch.relu with a Triton implementation for GPU tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return triton_relu(x)
        # Fallback to PyTorch implementation for CPU tensors
        return torch.relu(x)


# ---- Helper functions (kept from original code) ----
batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda")
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed