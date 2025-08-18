import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr,           # input tensor
    y_ptr,           # output tensor
    n_elements,      # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Swish: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig

    # Divide by 2
    y = y * 0.5

    # Clamp to [-1, 1]
    y = tl.maximum(y, -1.0)
    y = tl.minimum(y, 1.0)

    # Tanh
    exp2y = tl.exp(2.0 * y)
    y = (exp2y - 1.0) / (exp2y + 1.0)

    # Final clamp to [-1, 1]
    y = tl.maximum(y, -1.0)
    y = tl.minimum(y, 1.0)

    tl.store(y_ptr + offsets, y, mask=mask)


def fused_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model: keeps GEMM in cuBLAS but fuses all following element-wise ops
    (swish, divide, clamp, tanh, clamp) into one Triton kernel.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = self.gemm(x)
        x = fused_activation(x)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]


def get_init_inputs():
    return [in_features, out_features]