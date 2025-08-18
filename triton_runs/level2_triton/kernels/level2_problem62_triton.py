import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def leaky_relu_double_kernel(
    x_ptr,             # *pointer to input tensor
    out_ptr,           # *pointer to output tensor
    n_elements,        # total number of elements
    negative_slope,    # scalar float
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x >= 0, x * 2.0, x * negative_slope * 2.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_leaky_relu_double(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    Fused LeakyReLU + element-wise doubling implemented in Triton:
        y = leaky_relu(x, negative_slope)
        y = y + y
    """
    if not x.is_cuda:
        raise RuntimeError("Input tensor must be on CUDA for Triton kernel.")

    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    leaky_relu_double_kernel[grid](x, out, n_elements, negative_slope, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model where LeakyReLU and the subsequent x + x are fused
    into a single Triton kernel for higher throughput.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.negative_slope = negative_slope
        self.cuda()  # ensure parameters reside on GPU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.gn(x)
        x = triton_leaky_relu_double(x, self.negative_slope)
        return x


# --------------------------------------------------------------------
# Helper functions for the evaluation harness
batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512


def get_inputs():
    return [torch.rand(batch_size, input_size, device="cuda")]


def get_init_inputs():
    return [input_size, hidden_size, num_groups]