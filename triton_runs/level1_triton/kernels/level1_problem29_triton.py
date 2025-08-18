import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,         # *pointer* to input
    out_ptr,       # *pointer* to output
    n_elements,    # total number of elements
    beta,          # softplus beta
    threshold,     # softplus threshold
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    xb = x * beta
    # Compute stable softplus for xb
    z = xb
    soft = tl.where(z > 0, z, 0.0) + tl.log(1 + tl.exp(-tl.abs(z)))
    soft = soft / beta

    out = tl.where(xb > threshold, x, soft)

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0):
    """
    Applies Softplus activation using a custom Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    softplus_kernel[grid](x, out, n_elements, beta, threshold, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that applies Softplus activation using a custom Triton kernel.
    """
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        return triton_softplus(x, self.beta, self.threshold)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda")
    return [x]


def get_init_inputs():
    return []  # No special initialization required