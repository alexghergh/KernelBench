import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_div_kernel(
    x_ptr,          # *pointer* to the input tensor
    out_ptr,        # *pointer* to the output tensor
    div_val,        # scalar divisor  (float32)
    n_elements,     # total elements to process
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # ReLU
    x = tl.maximum(x, 0.0)
    # Divide by constant
    x = x / div_val
    # Store
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_relu_div(x: torch.Tensor, divisor: float):
    """
    Fuses ReLU + division by a scalar constant using a custom Triton kernel.
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    n_elements = x.numel()

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    relu_div_kernel[grid](x, out, divisor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps PyTorch's efficient Linear layer but
    fuses ReLU + division into a single custom Triton kernel.
    """
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = float(divisor)

    def forward(self, x):
        x = self.linear(x)                    # High-perf GEMM via cuBLAS
        x = triton_relu_div(x, self.divisor)  # Fused ReLU + division
        return x


# -------------------- helpers (unchanged interface) -------------------- #
batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]


def get_init_inputs():
    return [in_features, out_features, divisor]