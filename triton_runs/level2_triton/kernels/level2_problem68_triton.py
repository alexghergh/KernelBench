import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _min_sub_kernel(
    x_ptr,          # *f32
    out_ptr,        # *f32
    constant,       # f32
    n_elements,     # i32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x - constant
    out = tl.where(y > 0.0, 0.0, y)  # min(y, 0)
    tl.store(out_ptr + offsets, out, mask=mask)


def _triton_min_sub(x: torch.Tensor, constant: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and constant.is_cuda, "Inputs must be CUDA tensors."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _min_sub_kernel[grid](x, out, constant.item(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class _FusedMinSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, constant: torch.Tensor):
        out = _triton_min_sub(x, constant)
        ctx.save_for_backward(x, constant)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, constant = ctx.saved_tensors
        mask = (x <= constant).to(dtype=grad_out.dtype)
        grad_x = grad_out * mask
        grad_constant = (-grad_out * mask).sum()
        return grad_x, grad_constant


def fused_min_sub(x: torch.Tensor, constant: torch.Tensor) -> torch.Tensor:
    return _FusedMinSub.apply(x, constant)


class ModelNew(nn.Module):
    """
    Optimized model that fuses min and subtraction into a single Triton kernel.
    """
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant, dtype=torch.float32, device='cuda'))

    def forward(self, x):
        x = self.linear(x)
        x = fused_min_sub(x, self.constant)
        return x


# Required helper functions for the benchmarking harness
batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]


def get_init_inputs():
    return [in_features, out_features, constant]