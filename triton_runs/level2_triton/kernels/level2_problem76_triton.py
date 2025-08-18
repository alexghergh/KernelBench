import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bias_relu_kernel(
    x_ptr,          # *pointer* to the input matrix
    bias_ptr,       # *pointer* to the bias vector
    out_ptr,        # *pointer* to the output matrix
    M,              # number of rows in the input matrix
    N,              # number of columns / features
    BLOCK_SIZE: tl.constexpr,
):
    # Program/block indices
    pid_m = tl.program_id(0)        # row index
    pid_n = tl.program_id(1)        # column-block index

    # Column offsets handled by this program
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_n < N

    # Pointers to the row slice of x, bias, and out
    x_ptrs = x_ptr + pid_m * N + offs_n
    b_ptrs = bias_ptr + offs_n
    o_ptrs = out_ptr + pid_m * N + offs_n

    # Load data
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    # Fused bias addition + ReLU
    y = x + b
    y = tl.where(y > 0, y, 0)

    # Store result
    tl.store(o_ptrs, y, mask=mask)


def _triton_bias_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused bias addition and ReLU implemented with a Triton kernel.
    Args:
        x    : Tensor of shape (batch, features) on CUDA.
        bias : Tensor of shape (features,) on CUDA.
    Returns:
        out  : Tensor of the same shape as `x`.
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must reside on CUDA."
    assert x.shape[-1] == bias.shape[0], "Incompatible shapes."

    x = x.contiguous()
    bias = bias.contiguous()

    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = 128
    grid = (M, triton.cdiv(N, BLOCK_SIZE))

    bias_relu_kernel[grid](x, bias, out, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


class _BiasReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        ctx.save_for_backward(x, bias)
        return _triton_bias_relu(x, bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, bias = ctx.saved_tensors
        mask = (x + bias) > 0
        grad_in = grad_out.clone()
        grad_in *= mask
        grad_bias = grad_in.sum(dim=0)
        return grad_in, grad_bias


def bias_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Public wrapper exposing the fused op with autograd support."""
    return _BiasReLUFunction.apply(x, bias)


class ModelNew(nn.Module):
    """
    Optimized model:
    - Keeps the Linear matmul in PyTorch.
    - Fuses bias addition + ReLU through a custom Triton kernel.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.gemm(x)
        x = bias_relu(x, self.bias)
        return x


# ---- Helpers (retain original API) ----
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]


def get_init_inputs():
    return [in_features, out_features, bias_shape]