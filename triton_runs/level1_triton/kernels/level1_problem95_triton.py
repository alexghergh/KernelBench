import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr,      # *float32, shape: [BATCH, N_CLASS]
    targets_ptr,     # *int32,   shape: [BATCH]
    loss_ptr,        # *float32, shape: [BATCH]
    N_CLASS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each Triton program handles one example (one row)
    row_id = tl.program_id(0)

    # Pointer to the first element of the current row
    row_ptr = logits_ptr + row_id * N_CLASS

    # ---------------------------------------------------------------------
    # Pass 1: compute the maximum logit of the row for numerical stability
    # ---------------------------------------------------------------------
    offsets = tl.arange(0, BLOCK_SIZE_N)
    row_max = -float("inf")

    for start in tl.static_range(0, N_CLASS, BLOCK_SIZE_N):
        mask = offsets + start < N_CLASS
        x = tl.load(row_ptr + start + offsets, mask=mask, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    # ---------------------------------------------------------------------
    # Pass 2: compute sum(exp(logits - row_max))
    # ---------------------------------------------------------------------
    row_sum_exp = 0.0
    for start in tl.static_range(0, N_CLASS, BLOCK_SIZE_N):
        mask = offsets + start < N_CLASS
        x = tl.load(row_ptr + start + offsets, mask=mask, other=0.0)
        x = tl.exp(x - row_max)
        row_sum_exp += tl.sum(x, axis=0)

    # ---------------------------------------------------------------------
    # Gather the logit corresponding to the ground-truth class
    # ---------------------------------------------------------------------
    target_idx = tl.load(targets_ptr + row_id)
    target_idx = tl.cast(target_idx, tl.int32)
    logit_target = tl.load(row_ptr + target_idx)

    # ---------------------------------------------------------------------
    # Cross-entropy loss for the row:  −logit_target + log(sum_exp) + row_max
    # ---------------------------------------------------------------------
    loss_val = (-logit_target + row_max) + tl.log(row_sum_exp)

    # Store the per-sample loss
    tl.store(loss_ptr + row_id, loss_val)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    predictions : [batch, num_classes] (float32, CUDA)
    targets     : [batch]              (int64 / int32,  CUDA)
    Returns the mean cross-entropy loss (scalar tensor).
    """
    assert predictions.is_cuda, "Predictions must reside on CUDA device"
    assert targets.is_cuda, "Targets must reside on CUDA device"
    assert predictions.ndim == 2, "Expected predictions with shape [batch, num_classes]"

    predictions = predictions.contiguous()
    targets = targets.contiguous().int()        # Ensure int32 for Triton

    BATCH, N_CLASS = predictions.shape
    BLOCK_SIZE_N = 1024                         # Tunable; must divide N_CLASS
    assert N_CLASS % BLOCK_SIZE_N == 0, "N_CLASS must be divisible by BLOCK_SIZE_N"

    # Allocate output buffer for per-sample losses
    loss_buf = torch.empty(BATCH, device=predictions.device, dtype=predictions.dtype)

    grid = lambda meta: (BATCH,)

    cross_entropy_kernel[grid](
        predictions,
        targets,
        loss_buf,
        N_CLASS=N_CLASS,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return loss_buf.mean()


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel to compute
    the mean cross-entropy loss for multi-class classification.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)


# -----------------------------------------------------------------------------
# Helper functions for benchmarking / external usage (unchanged API)
# -----------------------------------------------------------------------------
batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)


def get_inputs():
    preds = torch.rand(batch_size, *input_shape, device="cuda", dtype=torch.float32)
    targs = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    return [preds, targs]


def get_init_inputs():
    return []