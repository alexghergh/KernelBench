import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    x_ptr,          # pointer to the input  tensor
    out_ptr,        # pointer to the output tensor
    rows: tl.int32, # number of rows    (batch_size)
    cols: tl.int32, # number of columns (dim)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that performs row-wise L1 normalisation:
        out[row, col] = x[row, col] / mean(|x[row, :]|)

    Each Triton program handles one row.
    Two passes over the row are performed inside the same kernel:
        1. compute the absolute-sum (to derive the mean)
        2. write the normalised values
    """
    row_id = tl.program_id(0)
    if row_id >= rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_start   = row_id * cols

    # ------------------------------------------------------------------
    # Pass 1 : accumulate sum(|x|)
    acc = tl.zeros((), dtype=tl.float32)

    offset = 0
    while offset < cols:
        offs  = offset + col_offsets
        mask  = offs < cols
        ptr   = x_ptr + row_start + offs
        vals  = tl.load(ptr, mask=mask, other=0.0)
        vals  = tl.abs(vals.to(tl.float32))
        vals  = tl.where(mask, vals, 0.0)
        acc  += tl.sum(vals, axis=0)
        offset += BLOCK_SIZE

    inv_mean = cols / acc  # (mean = acc / cols) -> reciprocal saves one div

    # ------------------------------------------------------------------
    # Pass 2 : write normalised values
    offset = 0
    while offset < cols:
        offs  = offset + col_offsets
        mask  = offs < cols
        ptr   = x_ptr + row_start + offs
        vals  = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        norm  = vals * inv_mean
        out_ptr_row = out_ptr + row_start + offs
        tl.store(out_ptr_row, norm, mask=mask)
        offset += BLOCK_SIZE


def triton_l1_normalize(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    Row-wise L1 normalisation implemented with Triton.
    Args:
        x (torch.Tensor): (batch_size, dim) CUDA tensor
        block_size (int): number of columns each thread-block processes at once
    Returns:
        torch.Tensor: L1-normalised tensor (same shape as x)
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    batch, dim = x.shape
    out = torch.empty_like(x)

    grid = lambda meta: (batch,)
    l1_norm_kernel[grid](x, out, batch, dim, BLOCK_SIZE=block_size)
    return out


class ModelNew(nn.Module):
    """
    Optimised version of the original model using a fused Triton kernel
    for row-wise L1 normalisation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton implementation when running on GPU, fall back to Torch on CPU
        if x.is_cuda:
            return triton_l1_normalize(x)
        else:
            return x / torch.mean(torch.abs(x), dim=1, keepdim=True)