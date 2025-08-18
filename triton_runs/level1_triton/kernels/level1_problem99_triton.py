import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_margin_kernel(
    anchor_ptr,           # pointer to anchor embeddings
    positive_ptr,         # pointer to positive embeddings
    negative_ptr,         # pointer to negative embeddings
    out_ptr,              # pointer to per-sample loss
    margin,               # margin (scalar float)
    N_DIM: tl.constexpr,  # embedding dimension (compile-time constant)
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)          # current sample index
    row_start = pid * N_DIM         # starting offset of this sample in the flat tensor

    acc_ap = tl.zeros((), tl.float32)  # accumulator for ||a - p||^2
    acc_an = tl.zeros((), tl.float32)  # accumulator for ||a - n||^2

    # iterate over the embedding dimension in chunks of BLOCK_SIZE
    for off in range(0, N_DIM, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        col_offsets = off + offsets
        mask = col_offsets < N_DIM

        idx = row_start + col_offsets

        a = tl.load(anchor_ptr + idx, mask=mask, other=0.0)
        p = tl.load(positive_ptr + idx, mask=mask, other=0.0)
        n = tl.load(negative_ptr + idx, mask=mask, other=0.0)

        diff_ap = a - p
        diff_an = a - n

        acc_ap += tl.sum(diff_ap * diff_ap, axis=0)
        acc_an += tl.sum(diff_an * diff_an, axis=0)

    d_ap = tl.sqrt(acc_ap)
    d_an = tl.sqrt(acc_an)

    loss = d_ap - d_an + margin
    loss = tl.where(loss > 0.0, loss, 0.0)

    tl.store(out_ptr + pid, loss)


def triton_triplet_margin_loss(anchor: torch.Tensor,
                               positive: torch.Tensor,
                               negative: torch.Tensor,
                               margin: float = 1.0,
                               block_size: int = 1024) -> torch.Tensor:
    """
    Computes Triplet Margin Loss using a fused Triton kernel.
    Returns a scalar tensor containing the mean loss.
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All inputs must be CUDA tensors."
    assert anchor.shape == positive.shape == negative.shape, "Input shapes must match."

    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()

    batch_size, n_dim = anchor.shape
    out = torch.empty(batch_size, device=anchor.device, dtype=anchor.dtype)

    grid = lambda meta: (batch_size,)

    triplet_margin_kernel[grid](
        anchor, positive, negative, out,
        float(margin),
        BLOCK_SIZE=block_size,
        N_DIM=n_dim,
        num_warps=4
    )

    return out.mean()


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for Triplet Margin Loss.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)


# ----------- utilities for the benchmarking framework -----------

batch_size = 32768
input_shape = (8192,)
dim = 1


def get_inputs():
    scale = torch.rand((), device='cuda')
    return [
        torch.rand(batch_size, *input_shape, device='cuda') * scale,
        torch.rand(batch_size, *input_shape, device='cuda'),
        torch.rand(batch_size, *input_shape, device='cuda'),
    ]


def get_init_inputs():
    return [1.0]  # default margin