import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_bias_tanh_kernel(
    in_ptr,            # Pointer to input tensor (after ConvTranspose2d)
    bias_ptr,          # Pointer to bias tensor (shape [C])
    out_ptr,           # Pointer to output tensor
    n_elements,        # Total number of elements in input/output
    stride_hw,         # H * W  (elements in one channel)
    C,                 # Number of channels
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index and load corresponding bias value
    channel_idx = ((offsets // stride_hw) % C)
    b = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)

    # Bias subtraction
    x = x - b

    # Fast tanh via exp: tanh(x) = (1 - e^{-2x}) / (1 + e^{-2x})
    exp_neg2x = tl.exp(-2.0 * x)
    y = (1.0 - exp_neg2x) / (1.0 + exp_neg2x)

    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_bias_tanh(input_tensor: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused (input - bias) followed by tanh implemented with Triton.
    `bias` is expected to be shape [C] or [C, 1, 1].
    """
    assert input_tensor.is_cuda and bias.is_cuda, "Tensors must be CUDA tensors"

    inp = input_tensor.contiguous()
    bias_flat = bias.view(-1).contiguous()

    N, C, H, W = inp.shape
    n_elements = inp.numel()
    stride_hw = H * W

    out = torch.empty_like(inp)

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_bias_tanh_kernel[grid](
        inp, bias_flat, out,
        n_elements,
        stride_hw,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model: keeps ConvTranspose2d from PyTorch, but fuses
    bias subtraction and tanh activation in a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape,
                 stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Extra bias parameter (same as original architecture)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_bias_tanh(x, self.bias)
        return x


# Helper functions (unchanged interface)
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]