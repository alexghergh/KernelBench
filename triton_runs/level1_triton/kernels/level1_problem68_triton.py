import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bias_add_kernel(
    out_ptr,        # *pointer* to output tensor
    bias_ptr,       # *pointer* to bias vector
    n_elements,     # total number of elements in the output tensor
    stride_c,       # stride between two channels in the flattened tensor
    C,              # number of channels
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load existing output values
    out_vals = tl.load(out_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index for every element
    chan_idx = (offsets // stride_c) % C
    bias_vals = tl.load(bias_ptr + chan_idx, mask=mask)

    # Add bias
    out_vals = out_vals + bias_vals

    # Store results
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def triton_add_bias(output: torch.Tensor, bias: torch.Tensor):
    """
    Adds `bias` (1-D tensor with size = out_channels) to `output`
    (5-D tensor [N, C, D, H, W]) using a Triton kernel.
    """
    assert output.is_cuda and bias.is_cuda, "Both tensors must be on CUDA."
    out = output.contiguous()
    bias = bias.contiguous()

    n_elements = out.numel()
    stride_c = out.stride(1)  # distance between two consecutive channels
    C = bias.shape[0]

    BLOCK_SIZE = 1024
    grid = lambda meta: (
        (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    bias_add_kernel[grid](out, bias, n_elements, stride_c, C, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the provided model.
    Performs a transposed 3D convolution (without bias) followed by a
    Triton-accelerated bias addition.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        # Disable bias inside ConvTranspose3d; we'll add it ourselves with Triton.
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_transpose3d(x)
        if self.bias is not None:
            out = triton_add_bias(out, self.bias)
        return out


# ---------------------------------------------------------------------
# Helper functions (kept identical signatures to original architecture)
# ---------------------------------------------------------------------
batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
height = 64
width = 64


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width).cuda()
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]