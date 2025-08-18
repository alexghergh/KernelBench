import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _clamp_mul_max_kernel(
    x_ptr,          # Pointer to the normalized tensor  (N, C, D, H, W)
    mult_ptr,       # Pointer to the multiplier vector (C,)
    out_ptr,        # Pointer to the output tensor     (N, D, H, W)
    N,              # Batch size
    DHW,            # D*H*W – spatial size
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    C: tl.constexpr,            # Number of channels (compile-time constant)
):
    pid = tl.program_id(0)      # Each program processes one (n, d, h, w) pixel
    n = pid // DHW              # Batch index
    s = pid % DHW               # Spatial index (flattened)

    base = n * C * DHW + s      # Offset of channel-0 element for this pixel

    # Load channel 0
    val = tl.load(x_ptr + base)
    val = tl.maximum(val, CLAMP_MIN)
    val = tl.minimum(val, CLAMP_MAX)
    scale = tl.load(mult_ptr + 0)
    val = val * scale
    max_val = val

    # Iterate over remaining channels
    for c in range(1, C):
        off = base + c * DHW
        val = tl.load(x_ptr + off)
        val = tl.maximum(val, CLAMP_MIN)
        val = tl.minimum(val, CLAMP_MAX)
        scale = tl.load(mult_ptr + c)
        val = val * scale
        max_val = tl.maximum(max_val, val)

    # Store result
    tl.store(out_ptr + pid, max_val)


def fused_clamp_mul_max(
    x: torch.Tensor,
    multiplier: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor:
    """
    Fuses torch.clamp + multiplication with `multiplier` +
    channel-wise max into a single Triton kernel.
    """
    assert x.is_cuda and multiplier.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float32, "Kernel supports float32 only"

    N, C, D, H, W = x.shape
    DHW = D * H * W

    x_contig = x.contiguous()
    mult_contig = multiplier.view(-1).contiguous()

    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (N * DHW,)

    _clamp_mul_max_kernel[grid](
        x_contig,
        mult_contig,
        out,
        N,
        DHW,
        CLAMP_MIN=clamp_min,
        CLAMP_MAX=clamp_max,
        C=C,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with a fused Triton kernel for
    clamp + multiply + channel-wise max.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        multiplier_shape,
        clamp_min,
        clamp_max,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier            # First multiplication
        x = self.instance_norm(x)          # Instance normalization
        # Fused clamp + second multiplication + max
        x = fused_clamp_mul_max(x, self.multiplier, self.clamp_min, self.clamp_max)
        return x