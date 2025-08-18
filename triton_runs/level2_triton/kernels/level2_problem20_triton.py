import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_bias_residual_mul_kernel(
    x_ptr,          # Pointer to the conv_transpose output
    bias_ptr,       # Pointer to the bias vector (C,)
    out_ptr,        # Pointer to output tensor
    n_elements,     # Total number of elements in x / out
    C,              # Number of channels
    DHW,            # D * H * W (volume of one channel for one batch elem)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the conv output
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Determine channel index for every element
    c_idx = ((offsets // DHW) % C).to(tl.int32)

    # Load corresponding bias value
    bias_val = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # Perform: x = x + bias; x = x + original_x; x = x * original_x; x = x + original_x
    # (original_x == x value–wise, but treated as constant in original model)
    tmp = x + bias_val      # x + bias
    tmp = tmp + x           # + original_x
    tmp = tmp * x           # * original_x
    out = tmp + x           # + original_x

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bias_residual_mul(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fuses: (x + bias) -> + original_x -> * original_x -> + original_x
    into a single Triton kernel.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert bias.is_cuda, "Bias must be on CUDA"

    x = x.contiguous()
    bias_flat = bias.contiguous().view(-1)  # (C,)
    out = torch.empty_like(x)

    n_elements = x.numel()
    C = x.shape[1]
    DHW = x.shape[2] * x.shape[3] * x.shape[4]  # Volume per channel

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_bias_residual_mul_kernel[grid](
        x, bias_flat, out, n_elements, C, DHW, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model: keeps the ConvTranspose3d layer from PyTorch,
    but fuses the subsequent bias add, residual adds, and multiply
    into a single custom Triton kernel for better memory-bandwidth
    utilization.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # bias shape expected as (C, 1, 1, 1) – keep identical to original
        self.bias = nn.Parameter(torch.randn(bias_shape).cuda())

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_bias_residual_mul(x, self.bias)
        return x