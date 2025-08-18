import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardsigmoid_kernel(
    x_ptr,            # *Pointer* to input tensor
    out_ptr,          # *Pointer* to output tensor
    n_elements,       # Number of elements in tensor
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    # Compute the offsets for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to guard memory accesses beyond the end of the tensor
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # HardSigmoid: relu6(x + 3) / 6
    x_f32 = x.to(tl.float32)
    y = x_f32 + 3.0
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 6.0)
    y = y / 6.0

    y = y.to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_hardsigmoid(x: torch.Tensor, *, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """
    Applies HardSigmoid using a custom Triton kernel.
    """
    assert x.is_cuda, "Input tensor must reside on GPU."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs HardSigmoid activation via a Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardsigmoid(x)


# ----- Helper functions (same signature as original) -----
batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda")
    return [x]


def get_init_inputs():
    return []