import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_sigmoid_scale_residual_kernel(
    x_ptr,          # pointer to input tensor
    out_ptr,        # pointer to output tensor
    scale,          # scaling factor (float32)
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))       # sigmoid(x)
    out = scale * sig + x                # scale * sigmoid(x) + x
    tl.store(out_ptr + offsets, out, mask=mask)


def _launch_fused_kernel(x: torch.Tensor, scale: float) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must reside on CUDA device."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _fused_sigmoid_scale_residual_kernel[grid](
        x, out, scale, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class _FusedSigmoidScaleResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        ctx.save_for_backward(input)
        return _launch_fused_kernel(input, scale)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input,) = ctx.saved_tensors
        scale = ctx.scale
        sig = torch.sigmoid(input)
        grad_input = grad_output * (scale * sig * (1.0 - sig) + 1.0)
        return grad_input, None


def fused_sigmoid_scale_residual(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _FusedSigmoidScaleResidualFunction.apply(x, scale)


class ModelNew(nn.Module):
    """
    Optimized model that fuses Sigmoid, Scaling and ResidualAdd into a single Triton kernel.
    """
    def __init__(self, input_size: int, hidden_size: int, scaling_factor: float):
        super().__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)
        x = fused_sigmoid_scale_residual(x, self.scaling_factor)
        return x


# --- Helper functions (kept identical to original API) -----------------------
batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, input_size, device="cuda")]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]