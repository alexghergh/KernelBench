import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_tanh_sub_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    subtract1,      # First scalar to subtract
    subtract2,      # Second scalar to subtract
    n_elements,     # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # x = tanh(x - subtract1) - subtract2
    x = x - subtract1
    exp_2x = tl.exp(2.0 * x)
    tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
    out = tanh_x - subtract2

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_tanh_sub(x: torch.Tensor, subtract1: float, subtract2: float) -> torch.Tensor:
    """
    Apply tanh(x - subtract1) - subtract2 element-wise using a Triton kernel.
    """
    assert x.is_cuda, "Input tensor must reside on CUDA device."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_tanh_sub_kernel[grid](x, out, subtract1, subtract2, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a convolution followed by a fused subtraction–tanh–subtraction
    implemented in Triton, and finally average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        x = self.conv(x)
        x = fused_tanh_sub(x, self.subtract1_value, self.subtract2_value)
        x = self.avgpool(x)
        return x


# ---------- Helper functions (unchanged API) ----------
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size,
            subtract1_value, subtract2_value, kernel_size_pool]