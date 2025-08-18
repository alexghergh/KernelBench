import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def subtract_mean_kernel(
    x_ptr,          # *fp32, full input tensor flattened
    mean_ptr,       # *fp32, (N*C) mean values
    out_ptr,        # *fp32, output tensor flattened
    S,              # int32, spatial size = D*H*W
    n_elements,     # int32, total number of elements in x/out
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean_idx = offs // S                       # which (N, C) this element belongs to
    mean_val = tl.load(mean_ptr + mean_idx, mask=mask, other=0.0)

    tl.store(out_ptr + offs, x - mean_val, mask=mask)


def triton_mean_subtract(x: torch.Tensor) -> torch.Tensor:
    """
    Subtract per-sample-channel mean from a 5-D tensor (N,C,D,H,W) with Triton.
    """
    assert x.dim() == 5 and x.is_cuda, "Input must be a CUDA tensor of shape (N,C,D,H,W)"
    x = x.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    n_elements = x.numel()

    # Per-(N,C) mean over spatial dimensions
    mean = x.mean(dim=(2, 3, 4)).contiguous()          # shape (N, C)
    mean_flat = mean.view(-1)                          # shape (N*C,)

    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    subtract_mean_kernel[grid](x, mean_flat, out, S, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d + BatchNorm3d with Triton-accelerated mean subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = triton_mean_subtract(x)
        return x


# --------------------------------------------------------------------
# Helper functions expected by the evaluation harness
# --------------------------------------------------------------------
batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]