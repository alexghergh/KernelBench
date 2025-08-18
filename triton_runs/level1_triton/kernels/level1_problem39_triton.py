import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2norm_kernel(
    x_ptr,        # * pointer to the input
    out_ptr,      # * pointer to the output
    n_cols,       #   number of features per row
    stride,       #   stride between consecutive rows (in elements)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_id = tl.program_id(0)

    # Offsets inside a row
    offs = tl.arange(0, BLOCK_SIZE)
    row_in_ptr = x_ptr + row_id * stride
    row_out_ptr = out_ptr + row_id * stride

    # -------- Pass 1: compute the squared L2-norm of the row --------
    sum_sq = tl.zeros((), dtype=tl.float32)
    offset = 0
    while offset < n_cols:
        cols = offset + offs
        mask = cols < n_cols
        x = tl.load(row_in_ptr + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
        offset += BLOCK_SIZE

    norm = tl.sqrt(sum_sq)
    inv_norm = 1.0 / norm  # no ε: follows torch.norm behaviour

    # -------- Pass 2: write normalized values --------
    offset = 0
    while offset < n_cols:
        cols = offset + offs
        mask = cols < n_cols
        x = tl.load(row_in_ptr + cols, mask=mask, other=0.0)
        y = x * inv_norm
        tl.store(row_out_ptr + cols, y, mask=mask)
        offset += BLOCK_SIZE


def triton_l2norm(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Applies L2 normalisation along `dim` using the custom Triton kernel.
    Supports 2-D tensors with `dim == 1`.
    """
    assert x.ndim == 2, "Only 2-D tensors are supported."
    assert dim == 1, "This implementation normalises along dim=1 only."

    original_device = x.device
    if not x.is_cuda:
        x = x.cuda()

    x = x.contiguous()
    batch, features = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024  # Tune for your GPU

    # One Triton program per row
    grid = lambda meta: (batch,)

    l2norm_kernel[grid](
        x,                       # x_ptr
        out,                     # out_ptr
        features,                # n_cols
        x.stride(0),             # stride between rows
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if original_device.type == "cpu":
        out = out.cpu()

    return out


class ModelNew(nn.Module):
    """
    Optimised model that performs L2 normalisation with a custom Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2norm(x, dim=1)


# Maintain original helper API
batch_size = 32768
dim = 65535


def get_inputs():
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]


def get_init_inputs():
    return []