import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr,          # Pointer to the input tensor
    out_ptr,        # Pointer to the output tensor
    n_elements,     # Number of elements to process
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_swish(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the Swish activation (x * sigmoid(x)) using a Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model with a fused Swish activation implemented in Triton.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_swish(x)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda")
    return [x]


def get_init_inputs():
    return []