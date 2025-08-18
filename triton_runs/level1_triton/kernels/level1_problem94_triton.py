import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 32768
input_shape = (32768,)
dim = 1  # Kept for compatibility


@triton.jit
def mse_partial_kernel(
    pred_ptr,          # pointer to predictions
    targ_ptr,          # pointer to targets
    partial_ptr,       # pointer to per-block partial sums
    n_elements,        # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    targ = tl.load(targ_ptr + offsets, mask=mask, other=0.0)
    diff = pred - targ
    sq   = diff * diff

    block_sum = tl.sum(sq, axis=0)
    tl.store(partial_ptr + pid, block_sum)


def triton_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes Mean Squared Error using a single-pass Triton kernel that fuses
    (pred - targ) ** 2 with a block-level reduction.
    """
    assert predictions.is_cuda and targets.is_cuda, "Inputs must be CUDA tensors."
    assert predictions.shape == targets.shape, "Shape mismatch between predictions and targets."

    predictions = predictions.contiguous()
    targets     = targets.contiguous()

    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    partial_sums = torch.empty(num_blocks, device=predictions.device,
                               dtype=predictions.dtype)

    mse_partial_kernel[(num_blocks,)](
        predictions, targets, partial_sums,
        n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    total_error = partial_sums.sum()
    return total_error / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)


def get_inputs():
    device = torch.device("cuda")
    scale = torch.rand((), device=device)
    return [
        torch.rand(batch_size, *input_shape, device=device) * scale,
        torch.rand(batch_size, *input_shape, device=device),
    ]


def get_init_inputs():
    return []