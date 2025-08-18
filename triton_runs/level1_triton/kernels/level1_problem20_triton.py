import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
#                          Triton LeakyReLU Kernel
# ----------------------------------------------------------------------
@triton.jit
def leaky_relu_kernel(
    x_ptr,           # * Pointer to the input tensor
    out_ptr,         # * Pointer to the output tensor
    alpha,           # * Negative slope (float32)
    n_elements,      # * Total number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # * Number of elements processed by each program
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x > 0, x, x * alpha)
    tl.store(out_ptr + offsets, y, mask=mask)

# ----------------------------------------------------------------------
#                     PyTorch ↔ Triton Wrapper Function
# ----------------------------------------------------------------------
def triton_leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    Applies LeakyReLU using a custom Triton kernel.

    Args:
        x (torch.Tensor): CUDA tensor.
        negative_slope (float): Slope for negative part.

    Returns:
        torch.Tensor: Result tensor (same shape/device as input).
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA to run the Triton kernel.")

    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    leaky_relu_kernel[grid](x, out, negative_slope, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out

# ----------------------------------------------------------------------
#                              New Model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that applies LeakyReLU with a custom Triton implementation.
    """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        return triton_leaky_relu(x, self.negative_slope)

# ----------------------------------------------------------------------
#                       Helper Functions for Runner
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393_216

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []