import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _scale_softmax_kernel(
    x_ptr,          # *f32
    out_ptr,        # *f32
    scale_ptr,      # *f32, single element
    stride_xm: tl.constexpr,
    n_cols,                             # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)              # row index
    offs = tl.arange(0, BLOCK_SIZE)     # column offsets [0, ..., BLOCK_SIZE)
    row_x_ptr = x_ptr + pid * stride_xm + offs
    row_out_ptr = out_ptr + pid * stride_xm + offs

    # Load scale (scalar)
    s = tl.load(scale_ptr)

    # Pass 1: compute max for numerical stability
    col_start = 0
    row_max = tl.full([1], -float('inf'), dtype=tl.float32)
    while col_start < n_cols:
        ptr = row_x_ptr + col_start
        mask = offs + col_start < n_cols
        x = tl.load(ptr, mask=mask, other=-float('inf'))
        x = x * s
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
        col_start += BLOCK_SIZE

    # Pass 2: compute exp(x - max) and row sum, write numerator to out
    col_start = 0
    row_sum = tl.full([1], 0.0, dtype=tl.float32)
    while col_start < n_cols:
        ptr = row_x_ptr + col_start
        mask = offs + col_start < n_cols
        x = tl.load(ptr, mask=mask, other=-float('inf'))
        x = x * s
        x = tl.exp(x - row_max)
        tl.store(row_out_ptr + col_start, x, mask=mask)
        row_sum += tl.sum(x, axis=0)
        col_start += BLOCK_SIZE

    inv_row_sum = 1.0 / row_sum

    # Pass 3: normalize
    col_start = 0
    while col_start < n_cols:
        ptr = row_out_ptr + col_start
        mask = offs + col_start < n_cols
        y = tl.load(ptr, mask=mask)
        y = y * inv_row_sum
        tl.store(ptr, y, mask=mask)
        col_start += BLOCK_SIZE


def _triton_scale_softmax(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Fused scale + softmax using Triton.
    Args:
        x     : (B, M) float32 tensor on CUDA
        scale : scalar tensor (shape [1]) on CUDA
    Returns:
        same shape as x
    """
    assert x.is_cuda and scale.is_cuda, "Inputs must be CUDA tensors"
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    n_rows, n_cols = x.shape
    grid = lambda meta: (n_rows,)

    _scale_softmax_kernel[grid](
        x, out, scale,
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class _ScaleSoftmaxAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        out = _triton_scale_softmax(x, scale)
        ctx.save_for_backward(out, x, scale)
        return out

    @staticmethod
    def backward(ctx, g_out):
        out, x, scale = ctx.saved_tensors
        # Gradient w.r.t. input
        dot = (g_out * out).sum(dim=1, keepdim=True)
        g_x = scale * out * (g_out - dot)
        # Gradient w.r.t. scale (scalar)
        g_scale = ((out * (g_out - dot)) * x).sum()
        return g_x, g_scale


def scale_softmax(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _ScaleSoftmaxAutograd.apply(x, scale)


class ModelNew(nn.Module):
    """
    Optimized model: keeps Linear + BatchNorm from PyTorch, fuses
    scaling and Softmax into a single Triton kernel.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bn_eps=1e-5,
        bn_momentum=0.1,
        scale_shape=(1,)
    ):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))

    def forward(self, x):
        x = self.gemm(x)
        x = self.bn(x)
        x = scale_softmax(x, self.scale)
        return x