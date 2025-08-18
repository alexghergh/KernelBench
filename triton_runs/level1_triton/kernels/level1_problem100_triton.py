import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_forward_kernel(
    preds_ptr,       # *float32
    targets_ptr,     # *float32
    out_ptr,         # *float32
    num_rows,        # int32
    row_size,        # int32
    num_elements,    # int32
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    preds = tl.load(preds_ptr + offsets, mask=mask, other=0.0)

    row_ids = offsets // row_size
    t = tl.load(targets_ptr + row_ids, mask=mask, other=0.0)

    hinge = 1.0 - preds * t
    hinge = tl.where(hinge > 0.0, hinge, 0.0)

    tl.store(out_ptr + offsets, hinge, mask=mask)


@triton.jit
def hinge_backward_kernel(
    preds_ptr,        # *float32
    targets_ptr,      # *float32
    grad_out_scalar,  # float32
    grad_pred_ptr,    # *float32
    num_rows,         # int32
    row_size,         # int32
    num_elements,     # int32
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    preds = tl.load(preds_ptr + offsets, mask=mask, other=0.0)

    row_ids = offsets // row_size
    t = tl.load(targets_ptr + row_ids, mask=mask, other=0.0)

    tmp = 1.0 - preds * t
    grad = tl.where(tmp > 0.0, -t, 0.0)
    grad = grad * grad_out_scalar / num_elements

    tl.store(grad_pred_ptr + offsets, grad, mask=mask)


class _TritonHingeLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions: torch.Tensor, targets: torch.Tensor):
        assert predictions.is_cuda and targets.is_cuda, "Inputs must be CUDA tensors."
        preds = predictions.contiguous()
        targs = targets.contiguous()

        num_rows = preds.shape[0]
        row_size = preds.numel() // num_rows
        num_elements = preds.numel()

        hinge_buf = torch.empty_like(preds)

        BLOCK_SIZE = 1024
        grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        hinge_forward_kernel[grid](
            preds, targs, hinge_buf,
            num_rows, row_size, num_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )

        loss = hinge_buf.mean()

        ctx.save_for_backward(preds, targs)
        ctx.num_elements = num_elements
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        preds, targs = ctx.saved_tensors
        num_elements = ctx.num_elements

        grad_preds = torch.empty_like(preds)

        num_rows = preds.shape[0]
        row_size = preds.numel() // num_rows

        BLOCK_SIZE = 1024
        grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        grad_scalar = grad_output.item()
        hinge_backward_kernel[grid](
            preds, targs, grad_scalar, grad_preds,
            num_rows, row_size, num_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # No gradient for targets (set to None)
        return grad_preds, None


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    return _TritonHingeLossFunction.apply(predictions, targets)


class ModelNew(nn.Module):
    """
    Hinge Loss model accelerated with custom Triton kernels.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions (unchanged from the original script)
# ----------------------------------------------------------------------------------------------------------------------
batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    preds = torch.rand(batch_size, *input_shape, device="cuda")
    targets = torch.randint(0, 2, (batch_size,), device="cuda").float() * 2 - 1
    return [preds, targets]


def get_init_inputs():
    return []