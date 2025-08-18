import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bn_scale_kernel(
    x_ptr,               # Input tensor
    mean_ptr,            # BatchNorm running mean
    var_ptr,             # BatchNorm running var
    weight_ptr,          # BatchNorm weight (gamma)
    bias_ptr,            # BatchNorm bias (beta)
    out_ptr,             # Output tensor
    n_elements,          # Total number of elements in x / out
    eps,                 # BatchNorm epsilon
    scale,               # Additional scaling factor
    NHW: tl.constexpr,   # H*W (spatial size)
    C: tl.constexpr,     # Number of channels
    BLOCK_SIZE: tl.constexpr,  # Number of threads per block
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Figure out which channel each element belongs to
    channel_idx = (offsets // NHW) % C

    # Load per-channel BatchNorm parameters
    mean = tl.load(mean_ptr + channel_idx, mask=mask)
    var = tl.load(var_ptr + channel_idx, mask=mask)
    w = tl.load(weight_ptr + channel_idx, mask=mask)
    b = tl.load(bias_ptr + channel_idx, mask=mask)

    inv_std = tl.math.rsqrt(var + eps)
    y = (x - mean) * inv_std * w + b
    y = y * scale

    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_batchnorm_scale_inference(x: torch.Tensor,
                                     bn: nn.BatchNorm2d,
                                     scaling_factor: float) -> torch.Tensor:
    """
    Fused BatchNorm (inference) + scalar scaling implemented in Triton.
    Falls back to PyTorch when bn.training is True.
    """
    assert x.is_cuda, "Input must be on CUDA."
    assert not bn.training, "Triton fusion supports inference mode only."

    device, dtype = x.device, x.dtype

    mean = bn.running_mean.to(device=device, dtype=dtype).contiguous()
    var = bn.running_var.to(device=device, dtype=dtype).contiguous()
    weight = bn.weight.to(device=device, dtype=dtype).contiguous()
    bias = bn.bias.to(device=device, dtype=dtype).contiguous()

    x = x.contiguous()
    N, C, H, W = x.shape
    NHW = H * W
    n_elements = x.numel()

    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    bn_scale_kernel[grid](
        x, mean, var, weight, bias, out,
        n_elements, bn.eps, scaling_factor,
        NHW=NHW, C=C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model where (BatchNorm2d + scalar scaling) is replaced
    by a single fused Triton kernel in inference mode.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        if self.training:
            # Standard PyTorch path for training
            x = self.bn(x)
            x = x * self.scaling_factor
        else:
            # Fused Triton path for inference
            x = triton_batchnorm_scale_inference(x, self.bn, self.scaling_factor)
        return x