import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_mish_tanh_kernel(
    x_ptr,            # *input*
    out_ptr,          # *output*
    n_elements,       # total number of elements
    BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    x = tl.load(x_ptr + idx, mask=mask, other=0.0)

    # softplus = log(1 + exp(x))
    exp_x = tl.exp(x)
    softplus = tl.log(1.0 + exp_x)

    # tanh(softplus)
    exp_neg2sp = tl.exp(-2.0 * softplus)
    tanh_sp = (1.0 - exp_neg2sp) / (1.0 + exp_neg2sp)

    mish = x * tanh_sp

    # tanh(mish)
    exp_neg2mish = tl.exp(-2.0 * mish)
    out_val = (1.0 - exp_neg2mish) / (1.0 + exp_neg2mish)

    tl.store(out_ptr + idx, out_val, mask=mask)


def fused_mish_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Applies fused Mish + Tanh activation to `x` using a custom Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_mish_tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    3D convolution followed by a fused Mish+Tanh activation implemented with Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )

    def forward(self, x):
        x = self.conv(x)
        x = fused_mish_tanh(x)
        return x


# -------------------------------------------------
# Helpers (same signature as original specification)
# -------------------------------------------------
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W, device='cuda')]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]