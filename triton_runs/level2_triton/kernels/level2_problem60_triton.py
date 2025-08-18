import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr,          # *pointer* to the input tensor
    out_ptr,        # *pointer* to the output tensor
    n_elements,     # total number of tensor elements
    BLOCK_SIZE: tl.constexpr,  # number of elements processed per block
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_swish(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elems = x.numel()

    grid = lambda meta: ((n_elems + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    swish_kernel[grid](x, out, n_elems, BLOCK_SIZE=block_size)
    return out


@triton.jit
def hardswish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tmp = x + 3.0
    tmp = tl.maximum(tmp, 0.0)
    tmp = tl.minimum(tmp, 6.0)
    y = x * (tmp / 6.0)

    tl.store(out_ptr + offsets, y, mask=mask)


def triton_hardswish(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elems = x.numel()

    grid = lambda meta: ((n_elems + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hardswish_kernel[grid](x, out, n_elems, BLOCK_SIZE=block_size)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that accelerates Swish and HardSwish activations using
    custom Triton kernels.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        eps,
        bias=True,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_swish(x)
        x = self.group_norm(x)
        x = triton_hardswish(x)
        return x


# ----- helpers expected by the evaluation harness -----
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device="cuda")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]