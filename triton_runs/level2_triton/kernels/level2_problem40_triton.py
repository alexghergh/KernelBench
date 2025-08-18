import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_residual_kernel(
    x_ptr,       # Pointer to the input tensor
    y_ptr,       # Pointer to the output tensor
    eff_scale,   # Effective scaling factor (1 + scaling_factor)
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * eff_scale
    tl.store(y_ptr + offsets, y, mask=mask)


def fused_scale_residual(x: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """
    Fuses scaling and residual addition into a single Triton kernel:
        y = x * (1 + scaling_factor)
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    eff_scale = 1.0 + scaling_factor
    scale_residual_kernel[grid](x, out, eff_scale, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a Triton kernel to fuse scaling and residual addition.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("scaling_factor", torch.tensor(scaling_factor, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = fused_scale_residual(x, float(self.scaling_factor))
        return x


# Default configuration values (kept from the original example)
batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5


def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda")]


def get_init_inputs():
    return [in_features, out_features, scaling_factor]