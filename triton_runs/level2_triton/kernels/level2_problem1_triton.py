import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel: fused ReLU + bias add
# ----------------------------------------------------------------------
@triton.jit
def relu_bias_kernel(
    x_ptr,          # float*  : input tensor
    bias_ptr,       # float*  : bias tensor (C, 1, 1)
    y_ptr,          # float*  : output tensor
    n_elements,     # int32   : total number of elements (N*C*H*W)
    C,              # int32   : number of channels
    HW,             # int32   : H*W (spatial size)
    BLOCK_SIZE: tl.constexpr  # meta-parameter: how many elements each program processes
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index for every element to fetch correct bias
    channel_idx = (offsets // HW) % C
    bias_val = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)

    # Fused op: ReLU + bias add
    y = tl.maximum(x, 0.0) + bias_val

    # Store
    tl.store(y_ptr + offsets, y, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper around the Triton kernel
# ----------------------------------------------------------------------
def triton_relu_bias(x: torch.Tensor, bias: torch.Tensor):
    """
    Apply fused ReLU + bias addition using a custom Triton kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    x = x.contiguous()
    bias = bias.contiguous()

    y = torch.empty_like(x)

    N, C, H, W = x.shape
    n_elements = x.numel()
    HW = H * W

    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    relu_bias_kernel[grid](x, bias, y,
                           n_elements, C, HW,
                           BLOCK_SIZE=BLOCK_SIZE)
    return y


# ----------------------------------------------------------------------
# Autograd-compatible wrapper
# ----------------------------------------------------------------------
class _ReLUBiasFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        ctx.save_for_backward(x, bias)
        return triton_relu_bias(x, bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, bias = ctx.saved_tensors

        grad_x = grad_out.clone()
        grad_x[x <= 0] = 0  # derivative of ReLU

        # grad for bias: sum over N, H, W
        grad_bias = grad_out.sum(dim=(0, 2, 3), keepdim=True)
        return grad_x, grad_bias


def fused_relu_bias(x, bias):
    return _ReLUBiasFunction.apply(x, bias)


# ----------------------------------------------------------------------
# Optimized model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Convolution followed by fused ReLU + bias add implemented in Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_relu_bias(x, self.bias)
        return x