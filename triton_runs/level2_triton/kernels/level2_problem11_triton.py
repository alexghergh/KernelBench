import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------- Triton kernel : BatchNorm (inference) + Tanh ------------------- #
@triton.jit
def bn_tanh_kernel(
    x_ptr,          # input
    w_ptr, b_ptr,   # weight (gamma) and bias (beta)
    mean_ptr, var_ptr,  # running mean / var
    out_ptr,        # output
    hw, C, NEL,     # spatial size (H*W), channel count, total element count
    eps,            # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off_start = pid * BLOCK_SIZE
    offs = off_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < NEL

    # load x
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # compute channel index for every element  (NCHW -> C major)
    hw_val = tl.full_like(offs, hw)
    c = (offs // hw_val) % C

    # gather per–channel parameters
    w   = tl.load(w_ptr   + c, mask=mask, other=1.0)
    b   = tl.load(b_ptr   + c, mask=mask, other=0.0)
    mu  = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr  + c, mask=mask, other=1.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (x - mu) * inv_std * w + b
    y = tl.tanh(y)

    tl.store(out_ptr + offs, y, mask=mask)


def triton_bn_tanh(x: torch.Tensor,
                   running_mean: torch.Tensor,
                   running_var: torch.Tensor,
                   weight: torch.Tensor,
                   bias: torch.Tensor,
                   eps: float = 1e-5):
    """
    Fused BatchNorm (inference) + tanh implemented in Triton.
    Only works in eval mode.
    """
    assert x.is_cuda, "Input must be on CUDA device."
    n, c, h, w = x.shape
    hw = h * w
    nel = x.numel()

    # fallbacks for None
    if weight is None:
        weight = torch.ones(c, device=x.device, dtype=x.dtype)
    if bias is None:
        bias = torch.zeros(c, device=x.device, dtype=x.dtype)

    # ensure contiguous
    x_ = x.contiguous()
    out = torch.empty_like(x_)

    BLOCK_SIZE = 1024
    grid = lambda meta: ((nel + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    bn_tanh_kernel[grid](
        x_, weight.contiguous(), bias.contiguous(),
        running_mean.contiguous(), running_var.contiguous(),
        out,
        hw, c, nel,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ------------------- Helper Module ------------------- #
class FusedBatchNormTanh(nn.Module):
    """
    Wrapper that uses the Triton fused kernel in eval mode,
    and falls back to PyTorch BatchNorm + tanh during training.
    """
    def __init__(self, bn_layer: nn.BatchNorm2d):
        super().__init__()
        self.bn = bn_layer

    def forward(self, x):
        if self.training:
            return torch.tanh(self.bn(x))
        # eval / inference path
        return triton_bn_tanh(
            x,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.bn.eps,
        )


# ------------------- Optimised Model ------------------- #
class ModelNew(nn.Module):
    """
    Same topology as the original model, but with a fused
    BatchNorm + tanh Triton implementation.
    """
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size, stride, padding,
                 groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding
        )

        # original BatchNorm layer – shared with fusion wrapper
        bn = nn.BatchNorm2d(out_channels)
        self.bn_tanh = FusedBatchNormTanh(bn)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups,
                                       num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn_tanh(x)          # fused BN + tanh
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x


# ------------------- Utility functions (same signatures) ------------------- #
batch_size = 512
in_channels = 64
out_channels = 128
height = width = 32
kernel_size = 5
stride = 1
padding = 1
groups = 8
num_groups = 8


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]