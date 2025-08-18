import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,          # *float32, input  pointer
    y_ptr,          # *float32, output pointer
    n_cols,         # int32,    number of columns (= features)
    BLOCK_SIZE: tl.constexpr,  # number of elements processed per kernel instance
):
    """
    Row–wise Softmax.

    Each Triton program handles one row (batch element) and iterates over the
    columns in chunks of BLOCK_SIZE.
    """
    row_id = tl.program_id(0)                       # which row this program is responsible for
    x_row_ptr = x_ptr + row_id * n_cols             # start of this row in input
    y_row_ptr = y_ptr + row_id * n_cols             # start of this row in output

    offs = tl.arange(0, BLOCK_SIZE)                 # [0 .. BLOCK_SIZE)
    row_max = -float("inf")                         # for numerical stability

    col = 0
    # 1) compute max
    while col < n_cols:
        mask = (col + offs) < n_cols
        x = tl.load(x_row_ptr + col + offs, mask=mask, other=-float("inf"))
        m = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, m)
        col += BLOCK_SIZE

    # 2) compute exps and their sum
    row_sum = 0.0
    col = 0
    while col < n_cols:
        mask = (col + offs) < n_cols
        x = tl.load(x_row_ptr + col + offs, mask=mask, other=-float("inf"))
        x_exp = tl.exp(x - row_max)
        tl.store(y_row_ptr + col + offs, x_exp, mask=mask)  # stash intermediate
        row_sum += tl.sum(x_exp, axis=0)
        col += BLOCK_SIZE

    # 3) normalize
    col = 0
    while col < n_cols:
        mask = (col + offs) < n_cols
        y = tl.load(y_row_ptr + col + offs, mask=mask)
        tl.store(y_row_ptr + col + offs, y / row_sum, mask=mask)
        col += BLOCK_SIZE


def triton_softmax(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Applies a row‐wise softmax (dim=1) using a custom Triton kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, features)
        dim (int):        Dimension to apply softmax on (must be 1)

    Returns:
        torch.Tensor: Softmax output, same shape as input.
    """
    assert dim == 1, "triton_softmax currently supports dim=1 only."
    assert x.dim() == 2, "Input must be 2-D (batch, features)."

    # Ensure we operate on GPU and contiguous memory
    x = x.contiguous().cuda()
    out = torch.empty_like(x)

    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024  # trade-off between occupancy and register pressure

    grid = lambda meta: (n_rows,)

    _softmax_kernel[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that replaces torch.softmax with a Triton implementation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x, dim=1)