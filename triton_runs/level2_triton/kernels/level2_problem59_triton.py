import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_scale_kernel(
    x_ptr,          # *pointer* to the input tensor
    out_ptr,        # *pointer* to the output tensor
    n_elements,     # total number of elements
    scaling_factor, # scalar multiplier
    BLOCK_SIZE: tl.constexpr,  # how many elements each program processes
):
    pid = tl.program_id(0)                      # unique program id
    block_start = pid * BLOCK_SIZE              # where the block starts
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # element ids processed by this program
    mask = offsets < n_elements                 # bounds check
    x = tl.load(x_ptr + offsets, mask=mask)     # load
    sig = 1.0 / (1.0 + tl.exp(-x))              # sigmoid
    y = x * sig * scaling_factor                # swish + scale
    tl.store(out_ptr + offsets, y, mask=mask)   # write back


def triton_swish_scale(x: torch.Tensor, scaling_factor: float):
    """
    Applies y = x * sigmoid(x) * scaling_factor using a Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    swish_scale_kernel[grid](
        x, out, n_elements, scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps the highly–tuned cuBLAS-backed Linear layer
    but replaces the Swish + scaling element-wise operations with a single
    fused Triton kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.matmul(x)
        x = triton_swish_scale(x, self.scaling_factor)
        return x