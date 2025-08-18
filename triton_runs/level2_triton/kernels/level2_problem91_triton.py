import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr,            # *float32,   [M, C]
    bias_ptr,         # *float32,   [C]
    out_ptr,          # *float32,   [M, C]
    M: tl.int32,      # rows  (N*H*W)
    C: tl.int32,      # cols  (out_channels)
    scaling: tl.float32,
    BLOCK_SIZE: tl.constexpr,   # must be >= C and power-of-2
):
    row = tl.program_id(0)               # each program handles one (n, h, w) location
    offs = tl.arange(0, BLOCK_SIZE)      # channel indices this program will process
    mask = offs < C

    row_ptr = x_ptr + row * C + offs     # pointers for this row

    x = tl.load(row_ptr, mask=mask, other=-1e30)

    # Softmax (numerically stable)
    max_val = tl.max(x, axis=0)
    x = x - max_val
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)
    softmax = exp_x / denom

    # Bias add, scale, sigmoid
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    y = (softmax + bias) * scaling
    y = 1.0 / (1.0 + tl.exp(-y))         # sigmoid

    tl.store(out_ptr + row * C + offs, y, mask=mask)


def fused_softmax_bias_scale_sigmoid(
    x: torch.Tensor, bias: torch.Tensor, scaling_factor: float
) -> torch.Tensor:
    """
    Expects x in NCHW.  Performs: softmax(dim=1) -> +bias -> *scale -> sigmoid
    in a single Triton kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"

    n, c, h, w = x.shape
    # Make channels the innermost dimension so they are contiguous
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    m = n * h * w                       # number of rows for the kernel
    x_2d = x_nhwc.view(m, c)            # [M, C]

    out = torch.empty_like(x_2d)
    bias_flat = bias.view(-1).contiguous()

    BLOCK_SIZE = triton.next_power_of_2(c)
    grid = lambda meta: (m,)

    fused_kernel[grid](x_2d, bias_flat, out, m, c, scaling_factor, BLOCK_SIZE=BLOCK_SIZE)

    out_nhwc = out.view(n, h, w, c)
    return out_nhwc.permute(0, 3, 1, 2).contiguous()


class ModelNew(nn.Module):
    """
    Transposed-conv followed by fused softmax + bias + scale + sigmoid.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        scaling_factor,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_softmax_bias_scale_sigmoid(x, self.bias, self.scaling_factor)
        return x


# ---------------------------------------------------------------------
# Helper functions expected by the evaluation harness
# ---------------------------------------------------------------------
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        scaling_factor,
    ]