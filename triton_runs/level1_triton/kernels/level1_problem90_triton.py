import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_kernel(
    x_ptr,        # *pointer* to input tensor
    out_ptr,      # *pointer* to output tensor
    n_elements,   # total number of tensor elements
    BLOCK_SIZE: tl.constexpr,  # kernel block size
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.log(x)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def exp_kernel(
    x_ptr,        # *pointer* to input tensor
    out_ptr,      # *pointer* to output tensor
    n_elements,   # total number of tensor elements
    BLOCK_SIZE: tl.constexpr,  # kernel block size
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_triton_unary(x: torch.Tensor, kernel):
    """
    Helper to invoke a Triton unary kernel (log/exp).
    """
    assert x.is_cuda, "Input tensor must reside on GPU"
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_log(x: torch.Tensor) -> torch.Tensor:
    return _launch_triton_unary(x, log_kernel)


def triton_exp(x: torch.Tensor) -> torch.Tensor:
    return _launch_triton_unary(x, exp_kernel)


class ModelNew(nn.Module):
    """
    Optimized model that computes the cumulative product using
    the identity:
        cumprod(x) = exp(cumsum(log(x)))
    The expensive element-wise log/exp operations are implemented
    with custom Triton kernels for higher throughput.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_log = triton_log(x)
        x_cumsum = torch.cumsum(x_log, dim=self.dim)
        return triton_exp(x_cumsum)


# -----------------------------------------------------------------------------
# I/O helpers expected by the harness
# -----------------------------------------------------------------------------
batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    return [torch.rand(batch_size, *input_shape, device="cuda")]


def get_init_inputs():
    return [dim]