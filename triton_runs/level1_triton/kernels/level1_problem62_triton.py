import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.ops as tops


def conv2d_triton(x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor = None,
                  stride=1,
                  padding=0,
                  dilation=1) -> torch.Tensor:
    """
    2-D convolution implemented as (im2col + Triton-accelerated matmul).
    Currently supports groups == 1.
    """
    assert x.is_cuda and weight.is_cuda, "All tensors must reside on CUDA"

    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, k_h, k_w = weight.shape

    # Normalise scalar arguments to 2-tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # ------------------------------------------------------------------
    # 1. im2col extraction (B, K, L) where K = in_c * k_h * k_w
    # ------------------------------------------------------------------
    patches = F.unfold(
        x,
        kernel_size=(k_h, k_w),
        dilation=dilation,
        padding=padding,
        stride=stride,           # shape: (B, K, L)
    )
    B, K, L = patches.shape

    # ------------------------------------------------------------------
    # 2. Triton‐backed matrix multiplication
    #    (out_channels, K)  @  (K, L)  →  (out_channels, L)
    # ------------------------------------------------------------------
    weight_mat = weight.view(out_channels, -1).contiguous()
    output = torch.empty(B, out_channels, L,
                         device=x.device,
                         dtype=x.dtype)

    for b in range(B):
        out_b = tops.matmul(weight_mat, patches[b])  # (M, L)
        if bias is not None:
            out_b += bias[:, None]
        output[b] = out_b

    # ------------------------------------------------------------------
    # 3. Reshape back to (B, C_out, H_out, W_out)
    # ------------------------------------------------------------------
    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    output = output.view(batch_size, out_channels, out_h, out_w)
    return output


class ModelNew(nn.Module):
    """
    Optimised 2-D convolution layer using Triton for the heavy GEMM.
    API-compatible (subset) with nn.Conv2d (groups==1).
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False):
        super().__init__()
        assert groups == 1, "Grouped convolution not supported in this Triton implementation."

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Parameters
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Kaiming-uniform initialisation (matches nn.Conv2d default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv2d_triton(x,
                             self.weight,
                             self.bias,
                             stride=self.stride,
                             padding=self.padding,
                             dilation=self.dilation)


# ----------------------------------------------------------------------
# Helper functions with the same signature as in the original file
# ----------------------------------------------------------------------
batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width, device="cuda")
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]