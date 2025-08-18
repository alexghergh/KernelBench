import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _scale_kernel(
    x_ptr,               # Pointer to the input matrix
    scale_ptr,           # Pointer to the scale vector
    out_ptr,             # Pointer to the output matrix
    n_elements,          # Total number of elements in the input matrix
    feature_dim,         # Number of columns (feature dimension)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    col_idx = offsets % feature_dim
    s = tl.load(scale_ptr + col_idx, mask=mask, other=0.0)

    tl.store(out_ptr + offsets, x * s, mask=mask)


def _triton_scale_impl(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Launches the Triton kernel that multiplies each feature in `x`
    by the corresponding element in `scale`.
    Shapes:
        x     : (batch, features)
        scale : (features,)
    """
    assert x.is_cuda and scale.is_cuda, "Both tensors must be on CUDA."
    x, scale = x.contiguous(), scale.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()
    feature_dim = x.shape[1]
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _scale_kernel[grid](x, scale, out, n_elements, feature_dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class _TritonScaleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor):
        ctx.save_for_backward(x, scale)
        return _triton_scale_impl(x, scale)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, scale = ctx.saved_tensors
        grad_x = _triton_scale_impl(grad_out, scale) if ctx.needs_input_grad[0] else None
        grad_scale = (grad_out * x).sum(dim=0) if ctx.needs_input_grad[1] else None
        return grad_x, grad_scale


def triton_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Element-wise scaling with autograd support powered by Triton.
    """
    return _TritonScaleFn.apply(x, scale)


class ModelNew(nn.Module):
    """
    Optimized model that replaces the element-wise scaling operation with
    a custom Triton kernel for higher performance.
    """
    def __init__(self, in_features, out_features, scale_shape,
                 eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(*scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)
        x = triton_scale(x, self.scale)
        x = self.bn(x)
        return x