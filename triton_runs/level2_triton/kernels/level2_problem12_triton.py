import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_leaky_relu_kernel(
    x_ptr,          # input tensor
    out_ptr,        # output tensor
    multiplier,     # scale factor
    negative_slope, # slope for LeakyReLU when x < 0
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x *= multiplier
    out = tl.where(x > 0, x, x * negative_slope)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scale_leaky_relu(x: torch.Tensor, multiplier: float, negative_slope: float):
    """
    Applies scaling and LeakyReLU in a single Triton kernel.
    """
    assert x.is_cuda, "Input must reside on CUDA device"

    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    scale_leaky_relu_kernel[grid](
        x, out,
        multiplier, negative_slope,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the original model.
    - Keeps the high-performance cuBLAS Linear layer.
    - Fuses multiplier and LeakyReLU into a single Triton kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        x = self.gemm(x)
        x = triton_scale_leaky_relu(x, self.multiplier, self.negative_slope)
        return x


# Helper functions (kept identical to original interface)
batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1


def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda")]


def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]