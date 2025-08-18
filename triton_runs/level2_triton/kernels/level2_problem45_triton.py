import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------- Triton Sigmoid -------------------------- #
@triton.jit
def _sigmoid_kernel(
    x_ptr,          # *f32
    out_ptr,        # *f32
    n_elements,     # i32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_sigmoid(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    Element-wise Sigmoid implemented with Triton.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)
    return out


# ----------------------- Triton LogSumExp (row) ---------------------- #
@triton.jit
def _logsumexp_kernel(
    x_ptr,          # *f32
    out_ptr,        # *f32
    stride_row,     # i32
    n_cols,         # i32
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row_ptr = x_ptr + row_id * stride_row + offsets
    x = tl.load(row_ptr, mask=mask, other=-float("inf"))

    row_max = tl.max(x, axis=0)
    x = x - row_max
    exp_x = tl.exp(x)
    row_sum = tl.sum(exp_x, axis=0)

    lse = row_max + tl.log(row_sum)
    tl.store(out_ptr + row_id, lse)


def triton_logsumexp(x: torch.Tensor, dim: int = 1, block_size: int = 1024) -> torch.Tensor:
    """
    Computes LogSumExp over `dim==1` for a 2-D tensor using Triton.
    """
    assert dim == 1, "Only dim=1 is supported."
    assert x.dim() == 2, "Input must be a 2-D tensor."
    assert x.is_cuda, "Input tensor must be on CUDA device."

    x = x.contiguous()
    batch, n_cols = x.shape
    out = torch.empty(batch, device=x.device, dtype=x.dtype)

    stride_row = x.stride(0)
    grid = (batch,)

    _logsumexp_kernel[grid](x, out, stride_row, n_cols, BLOCK_SIZE=block_size)
    return out


# ------------------------------ Model -------------------------------- #
class ModelNew(nn.Module):
    """
    Optimized version of the original model that replaces
    Sigmoid and LogSumExp with Triton kernels.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size).cuda()
        self.linear2 = nn.Linear(hidden_size, output_size).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.cuda()
        x = self.linear1(x)
        x = triton_sigmoid(x)
        x = self.linear2(x)
        x = triton_logsumexp(x, dim=1)
        return x


# ----------------------- Helper: input creators ---------------------- #
batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024


def get_inputs():
    return [torch.rand(batch_size, input_size, device="cuda")]


def get_init_inputs():
    return [input_size, hidden_size, output_size]