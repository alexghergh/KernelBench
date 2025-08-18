import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _row_softmax_kernel(
    input_ptr,          # fp16*
    output_ptr,         # fp16*
    n_cols,             # int32
    row_stride,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # One program = one row
    row_id = tl.program_id(0)

    # Pointers to the beginning of the current row
    row_in_ptr  = input_ptr  + row_id * row_stride
    row_out_ptr = output_ptr + row_id * row_stride

    # Column indices handled by this program
    offs  = tl.arange(0, BLOCK_SIZE)
    mask  = offs < n_cols

    # Load, cast to fp32 for stability
    x = tl.load(row_in_ptr + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # Numerically-stable softmax
    x_max = tl.max(x, axis=0)
    x     = x - x_max
    num   = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y     = (num / denom).to(tl.float16)

    # Store
    tl.store(row_out_ptr + offs, y, mask=mask)


def _triton_row_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax over the last dimension using a Triton kernel.
    Supports fp16 tensors that are contiguous in memory.
    """
    assert x.is_cuda, "Tensor must be on CUDA"
    assert x.dtype == torch.float16, "Only fp16 supported"

    x_contig = x.contiguous()
    out      = torch.empty_like(x_contig)

    n_cols     = x_contig.shape[-1]
    n_rows     = x_contig.numel() // n_cols
    row_stride = n_cols

    # Next power-of-two ≤ 1024 for BLOCK_SIZE
    block_size = 1 << int(math.ceil(math.log2(n_cols)))
    block_size = min(block_size, 1024)

    grid = lambda meta: (n_rows,)

    _row_softmax_kernel[grid](
        x_contig, out,
        n_cols, row_stride,
        BLOCK_SIZE=block_size,
    )

    return out.view_as(x_contig)


class ModelNew(nn.Module):
    """
    Optimized attention using a Triton-based softmax.
    """
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Q, K, V : [B, H, S, D]
        B, H, S, D = Q.shape
        scale = D ** -0.5

        # 1. Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, S, S]

        # 2. Softmax with Triton kernel
        probs = _triton_row_softmax(scores)

        # 3. Weighted value aggregation
        out = torch.matmul(probs, V)                           # [B, H, S, D]
        return out