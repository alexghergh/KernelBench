import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_kernel(
    pred_ptr,               # pointer to predictions
    target_ptr,             # pointer to targets
    partial_ptr,            # pointer to per-block partial sums
    n_elements,             # total number of elements
    BETA: tl.constexpr,     # smooth-L1 beta (threshold)
    BLOCK_SIZE: tl.constexpr,  # number of elements handled by each program
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load and up-cast to fp32 for stable math
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    diff = pred - target
    abs_diff = tl.abs(diff)

    loss_small = 0.5 * diff * diff / BETA
    loss_large = abs_diff - 0.5 * BETA
    loss = tl.where(abs_diff < BETA, loss_small, loss_large)

    block_sum = tl.sum(loss, axis=0)
    tl.store(partial_ptr + pid, block_sum)


def _triton_smooth_l1_mean(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0):
    """
    Fast Smooth-L1 (Huber) loss with reduction='mean' implemented in Triton.
    Returns a scalar tensor.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Shape mismatch between predictions and targets."

    predictions = predictions.contiguous()
    targets = targets.contiguous()

    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # accumulate partial sums in fp32 for accuracy
    partial_sums = torch.empty(grid[0], device=predictions.device, dtype=torch.float32)

    smooth_l1_kernel[grid](
        predictions, targets, partial_sums,
        n_elements,
        BETA=beta,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    total_loss = partial_sums.sum()
    mean_loss = total_loss / n_elements
    return mean_loss


class _SmoothL1LossTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets, beta=1.0):
        ctx.beta = beta
        ctx.save_for_backward(predictions, targets)
        return _triton_smooth_l1_mean(predictions, targets, beta)

    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        beta = ctx.beta

        diff = predictions - targets
        abs_diff = diff.abs()
        grad_pred = torch.where(abs_diff < beta, diff / beta, diff.sign())
        grad_pred = grad_pred * (grad_output / predictions.numel())
        grad_target = -grad_pred
        return grad_pred, grad_target, None


class ModelNew(nn.Module):
    """
    Optimized model computing Smooth L1 (Huber) Loss using a custom Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return _SmoothL1LossTritonFn.apply(predictions, targets, 1.0)