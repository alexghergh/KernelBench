import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_bn_gelu_relu_kernel(
    x_ptr,             # [M, N] input
    mean_ptr,          # [N] running mean
    var_ptr,           # [N] running var
    gamma_ptr,         # [N] weight (scale)
    beta_ptr,          # [N] bias (shift)
    out_ptr,           # [M, N] output
    M,                 # batch dimension
    N,                 # feature dimension
    EPS: tl.constexpr,               # batch-norm epsilon (compile-time constant)
    BLOCK_SIZE_N: tl.constexpr = 128 # columns processed per program
):
    """
    Fuses BatchNorm (inference mode) + GELU (tanh approximation) + ReLU.
    Grid  = (M, ceil_div(N, BLOCK_SIZE_N))
    """
    pid_m = tl.program_id(0)                     # row index
    pid_n = tl.program_id(1)                     # block index along columns

    col_start = pid_n * BLOCK_SIZE_N
    offs_n = col_start + tl.arange(0, BLOCK_SIZE_N)
    mask = offs_n < N

    row_offset = pid_m * N
    ptrs = row_offset + offs_n                  # row-major offset

    # Load data
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offs_n, mask=mask, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offs_n, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask, other=0.0)

    # BatchNorm (inference)
    inv_std = 1.0 / tl.sqrt(var + EPS)
    y = (x - mean) * inv_std * gamma + beta

    # GELU (tanh approximation)
    k = 0.7978845608          # sqrt(2/pi)
    c = 0.044715
    inner = k * (y + c * y * y * y)
    gelu = 0.5 * y * (1.0 + tl.tanh(inner))

    # ReLU
    out = tl.maximum(gelu, 0.0)

    # Store
    tl.store(out_ptr + ptrs, out, mask=mask)


def fused_bn_gelu_relu(x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
    """
    Applies BatchNorm (eval mode) + GELU + ReLU using a Triton kernel.
    Falls back to PyTorch when bn is in training mode.
    """
    if bn.training:
        # Fallback path (rarely used during inference benchmarking)
        return torch.relu(torch.nn.functional.gelu(bn(x)))

    assert x.is_cuda, "Input must be on CUDA device"
    x = x.contiguous()

    M, N = x.shape
    out = torch.empty_like(x)

    # Ensure parameters are on the same device and contiguous
    mean = bn.running_mean.to(x.device).contiguous()
    var = bn.running_var.to(x.device).contiguous()
    gamma = bn.weight.to(x.device).contiguous()
    beta = bn.bias.to(x.device).contiguous()

    BLOCK_SIZE_N = 128
    grid = (M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)

    fused_bn_gelu_relu_kernel[grid](
        x, mean, var, gamma, beta, out,
        M, N,
        EPS=bn.eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      (Linear) GEMM via cuBLAS
        -> fused BatchNorm (eval) + GELU + ReLU via Triton
    For training, falls back to the standard PyTorch ops to keep correctness.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.gemm(x)
        if self.training:
            # Maintain PyTorch semantics during training
            x = self.batch_norm(x)
            x = torch.nn.functional.gelu(x)
            x = torch.relu(x)
            return x
        else:
            # Fast fused inference path
            return fused_bn_gelu_relu(x, self.batch_norm)


# --------- helper functions (required by the harness) ----------
batch_size = 16384
in_features = 4096
out_features = 4096


def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda")]


def get_init_inputs():
    return [in_features, out_features]