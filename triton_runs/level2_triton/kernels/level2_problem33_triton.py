import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_kernel(
    x_ptr,          # input matrix
    scale_ptr,      # per-feature scale vector
    out_ptr,        # output matrix
    n_elements,     # total number of elements in the matrix
    n_cols,         # number of columns (= features)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    cols = offsets % n_cols
    s = tl.load(scale_ptr + cols, mask=mask, other=0.0)

    tl.store(out_ptr + offsets, x * s, mask=mask)


def triton_scale(x: torch.Tensor, scale: torch.Tensor):
    """
    Element-wise scaling with Triton. Supports any (batch, features) 2-D tensor.
    """
    assert x.is_cuda and scale.is_cuda, "Inputs must be on CUDA devices."
    assert x.dtype == scale.dtype, "dtype mismatch between tensor and scale."

    x = x.contiguous()
    scale = scale.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    n_cols = x.shape[-1]
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    scale_kernel[grid](x, scale, out, n_elements, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class _TritonScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x, scale)
        return triton_scale(x, scale)

    @staticmethod
    def backward(ctx, grad_out):
        x, scale = ctx.saved_tensors
        grad_x = grad_out * scale
        grad_scale = (grad_out * x).sum(0)
        return grad_x, grad_scale


def triton_mul(x, scale):
    return _TritonScaleFunction.apply(x, scale)


class ModelNew(nn.Module):
    """
    Optimized version of the original model.
    - Keeps PyTorch Linear and BatchNorm.
    - Replaces the per-feature scaling with a custom Triton kernel.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)               # GEMM
        x = triton_mul(x, self.scale)  # Triton-accelerated scaling
        x = self.bn(x)                 # BatchNorm
        return x


# ----------------------------------------------------------------------
# Helpers to match the required interface
# ----------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
scale_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]


def get_init_inputs():
    return [in_features, out_features, scale_shape]