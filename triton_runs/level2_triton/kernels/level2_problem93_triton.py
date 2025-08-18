import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_add_clamp_gelu_scale_kernel(
    x_ptr,                 # input tensor
    out_ptr,               # output tensor
    add_value,             # scalar to add
    multiply_value,        # scalar to multiply
    n_elements,            # total number of elements
    BLOCK_SIZE: tl.constexpr,  # block size
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # x = gelu(min(x + add_value, 0.0)) * multiply_value
    x = x + add_value
    x = tl.minimum(x, 0.0)

    INV_SQRT2 = 0.7071067811865476  # 1 / sqrt(2)
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))  # GELU

    x = x * multiply_value

    # Store
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_add_clamp_gelu_scale(
    x: torch.Tensor,
    add_value: float,
    multiply_value: float,
) -> torch.Tensor:
    """
    Applies: gelu(min(x + add_value, 0)) * multiply_value
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_add_clamp_gelu_scale_kernel[grid](
        x,
        out,
        add_value,
        multiply_value,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with a fused Triton kernel for
    add + clamp_max(0) + GELU + multiply.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        add_value,
        multiply_value,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
        )
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_add_clamp_gelu_scale(x, self.add_value, self.multiply_value)
        return x


# Helper functions expected by the environment
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0


def get_inputs():
    return [
        torch.rand(
            batch_size,
            in_channels,
            height,
            width,
            device="cuda",
        )
    ]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        add_value,
        multiply_value,
    ]