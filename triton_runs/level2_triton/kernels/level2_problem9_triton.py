import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_sub_mul_relu_kernel(
    inp_ptr,                        # pointer to input tensor
    out_ptr,                        # pointer to output tensor
    subtract_value,                 # scalar to subtract
    multiply_value,                 # scalar to multiply
    n_elements,                     # total number of elements
    BLOCK_SIZE: tl.constexpr,       # block size
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    x = (x - subtract_value) * multiply_value
    x = tl.where(x > 0, x, 0.0)      # ReLU
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_sub_mul_relu(x: torch.Tensor, subtract_value: float, multiply_value: float):
    """
    Launches a Triton kernel that fuses: (x - subtract_value) * multiply_value followed by ReLU.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elems = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elems + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_sub_mul_relu_kernel[grid](
        x, out,
        subtract_value,
        multiply_value,
        n_elems,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with a Triton kernel that fuses subtraction, multiplication and ReLU.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = float(subtract_value)
        self.multiply_value = float(multiply_value)

    def forward(self, x):
        x = self.linear(x)
        x = fused_sub_mul_relu(x, self.subtract_value, self.multiply_value)
        return x


# ----- The following helpers are required by the evaluation harness -----
batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5


def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]


def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]