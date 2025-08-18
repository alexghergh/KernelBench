import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,        # pointer to input tensor
    y_ptr,        # pointer to output tensor
    C, H, W,      # tensor dimensions
    eps,          # epsilon for numerical stability
    stride_b,     # strides to reach next batch
    stride_c,     # strides to reach next channel
    stride_h,     # strides to reach next row (H)
    stride_w,     # strides to reach next column (W)
    BLOCK_SIZE: tl.constexpr,  # number of channels processed per program
):
    pid = tl.program_id(0)             # unique program id
    w = pid % W                        # column index
    tmp = pid // W
    h = tmp % H                        # row index
    b = tmp // H                       # batch index

    # base address of the (b, 0, h, w) element
    base_offset = b * stride_b + h * stride_h + w * stride_w

    # offsets for the CHANNEL dimension
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C                    # mask to stay within valid channels
    ptrs = base_offset + offs * stride_c

    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0)

    mean_sq = tl.sum(x * x, axis=0) / C
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    y = x * inv_rms
    tl.store(y_ptr + ptrs, y, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Applies RMS Normalization using a custom Triton kernel.
    Args:
        x  : CUDA tensor of shape (B, C, H, W) in NCHW layout.
        eps: Epsilon for numerical stability.
    Returns:
        y  : Normalized tensor, same shape as x.
    """
    assert x.is_cuda and x.dtype == torch.float32, "x must be a CUDA float32 tensor"
    x = x.contiguous()

    B, C, H, W = x.shape
    y = torch.empty_like(x)

    # compute strides for NCHW contiguous tensor
    stride_w = 1
    stride_h = W
    stride_c = H * W
    stride_b = C * H * W

    BLOCK_SIZE = triton.next_power_of_2(C)
    grid = (B * H * W,)

    rmsnorm_kernel[grid](
        x, y,
        C, H, W,
        eps,
        stride_b,
        stride_c,
        stride_h,
        stride_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using a Triton kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.eps)


# Utilities to mimic the original interface
batch_size = 112
features = 64
dim1 = 512
dim2 = 512


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2, device='cuda')
    return [x]


def get_init_inputs():
    return [features]