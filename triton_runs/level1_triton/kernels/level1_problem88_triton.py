import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------ Triton GELU ------------------ #
@triton.jit
def gelu_kernel(
    x_ptr,          # *fp32, input
    out_ptr,        # *fp32, output
    n_elements,     # int32, total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Constants (fp32)
    SQRT_2_OVER_PI = 0.7978845608028654   # sqrt(2/pi)
    COEFF = 0.044715
    HALF = 0.5

    x_cube = x * x * x
    inner = x + COEFF * x_cube
    u = SQRT_2_OVER_PI * inner

    # tanh approximation using exp, avoids relying on intrinsic tanh
    e = tl.exp(2.0 * u)
    tanh_u = (e - 1.0) / (e + 1.0)

    gelu = HALF * x * (1.0 + tanh_u)
    tl.store(out_ptr + offsets, gelu, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must reside on CUDA device."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ------------------ Model ------------------ #
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return triton_gelu(x)


# ------------------ Input helpers ------------------ #
batch_size = 8192
dim = 8192


def get_inputs():
    return [torch.rand(batch_size, dim, device='cuda')]


def get_init_inputs():
    return []