import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel that computes per–row KL-divergence contributions
# -------------------------------------------------------------------------
@triton.jit
def kl_div_kernel(
    pred_ptr,          # *f32  : predictions (probabilities)
    targ_ptr,          # *f32  : targets      (probabilities)
    row_sum_ptr,       # *f32  : output buffer (one scalar per row)
    D,                 # int32 : number of columns / features
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(axis=0)      # batch dimension
    block_id = tl.program_id(axis=1)    # feature–tile index

    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    pred = tl.load(pred_ptr + row_id * D + offs, mask=mask, other=1.0)
    targ = tl.load(targ_ptr + row_id * D + offs, mask=mask, other=0.0)

    eps = 1e-10
    pred = tl.where(pred > eps, pred, eps)
    targ = tl.where(targ > eps, targ, eps)

    contrib = targ * (tl.log(targ) - tl.log(pred))
    partial_sum = tl.sum(contrib, axis=0)

    tl.atomic_add(row_sum_ptr + row_id, partial_sum)


# -------------------------------------------------------------------------
# Python wrapper around the Triton kernel
# -------------------------------------------------------------------------
def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean'),
    but computed with a fused Triton kernel for better performance.
    """
    assert predictions.is_cuda and targets.is_cuda, "Inputs must be on CUDA."
    assert predictions.shape == targets.shape, "predictions and targets must have the same shape."

    predictions = predictions.contiguous()
    targets = targets.contiguous()

    B, D = predictions.shape
    row_sums = torch.zeros((B,), dtype=predictions.dtype, device=predictions.device)

    BLOCK_SIZE = 256  # Tune for your GPU
    grid = (B, (D + BLOCK_SIZE - 1) // BLOCK_SIZE)

    kl_div_kernel[grid](predictions, targets, row_sums, D, BLOCK_SIZE=BLOCK_SIZE)
    return row_sums.mean()


# -------------------------------------------------------------------------
# Optimized model
# -------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)


# -------------------------------------------------------------------------
# Input helpers (unchanged from original)
# -------------------------------------------------------------------------
batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1


def get_inputs():
    scale = torch.rand(())
    return [
        (torch.rand(batch_size, *input_shape, device="cuda") * scale).softmax(dim=-1),
        torch.rand(batch_size, *input_shape, device="cuda").softmax(dim=-1),
    ]


def get_init_inputs():
    return []