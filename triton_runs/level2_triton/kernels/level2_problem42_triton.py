import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def logsumexp_bias_kernel(
    x_ptr,          # Pointer to input tensor  (B, C)
    bias_ptr,       # Pointer to bias tensor  (C,)
    out_ptr,        # Pointer to output tensor (B,)
    C,              # Number of channels
    BLOCK_SIZE: tl.constexpr,  # One wavefront handles BLOCK_SIZE channels
):
    b_id = tl.program_id(0)                       # Batch index
    offs = tl.arange(0, BLOCK_SIZE)               # Channel offsets handled by this wave
    mask = offs < C                               # Mask in case C < BLOCK_SIZE

    # Load inputs
    x = tl.load(x_ptr + b_id * C + offs, mask=mask, other=-1.0e30)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)

    v = x + bias                                  # Add bias
    m = tl.max(v, axis=0)                         # Max for numerical stability
    sum_exp = tl.sum(tl.exp(v - m), axis=0)       # Stable exp-sum
    res = (m + tl.log(sum_exp)) * 10.0            # logsumexp and scale

    tl.store(out_ptr + b_id, res)                 # Write result


def triton_logsumexp_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x    : (B, C)  CUDA tensor
    bias : (C,)    CUDA tensor
    returns (B, 1) CUDA tensor with 10 * logsumexp(x + bias, dim=1)
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    B, C = x.shape

    # Choose power-of-two BLOCK_SIZE ≥ C (capped at 1024 for register pressure)
    BLOCK_SIZE = 1
    while BLOCK_SIZE < C:
        BLOCK_SIZE <<= 1
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    out = torch.empty(B, device=x.device, dtype=x.dtype)

    grid = (B,)                                   # One program per batch element
    logsumexp_bias_kernel[grid](x, bias, out, C, BLOCK_SIZE=BLOCK_SIZE)

    return out.unsqueeze(1)                       # (B, 1)


class ModelNew(nn.Module):
    """
    Optimized model: native ConvTranspose2d, Triton-powered
    bias + logsumexp + scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)                # (B, C, H, W)
        x = torch.mean(x, dim=(2, 3))             # Global average pool -> (B, C)
        out = triton_logsumexp_bias(x, self.bias.view(-1))
        return out                                # (B, 1)