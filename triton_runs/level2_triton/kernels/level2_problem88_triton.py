import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_groupnorm_swish_kernel(
    x_ptr,               # [N, C] input
    gamma_ptr,           # [C]   GroupNorm weight
    beta_ptr,            # [C]   GroupNorm bias
    mul_w_ptr,           # [C]   element-wise multiply weight
    out_ptr,             # [N, C] output
    eps,                 # float, numeric stabilizer
    C: tl.constexpr,     # total channels
    G_SIZE: tl.constexpr # channels per group
):
    pid = tl.program_id(0)
    NUM_GROUPS = C // G_SIZE

    n = pid // NUM_GROUPS              # sample index
    g = pid %  NUM_GROUPS              # group index

    ch_offset   = g * G_SIZE + tl.arange(0, G_SIZE)
    ch_mask     = ch_offset < C
    row_offset  = n * C + ch_offset

    # load inputs
    x     = tl.load(x_ptr     + row_offset, mask=ch_mask, other=0.0)
    gamma = tl.load(gamma_ptr + ch_offset,  mask=ch_mask, other=0.0)
    beta  = tl.load(beta_ptr  + ch_offset,  mask=ch_mask, other=0.0)
    mul_w = tl.load(mul_w_ptr + ch_offset,  mask=ch_mask, other=0.0)

    # GroupNorm statistics
    mean = tl.sum(x, axis=0) / G_SIZE
    var  = tl.sum((x - mean) * (x - mean), axis=0) / G_SIZE
    inv_std = tl.math.rsqrt(var + eps)

    # GroupNorm affine
    y = (x - mean) * inv_std
    y = y * gamma + beta

    # Swish, multiply, Swish (all fused)
    sig1   = 1.0 / (1.0 + tl.exp(-y))
    inter  = y * sig1 * mul_w
    sig2   = 1.0 / (1.0 + tl.exp(-inter))
    out    = inter * sig2

    # write back
    tl.store(out_ptr + row_offset, out, mask=ch_mask)


def fused_groupnorm_swish(
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        num_groups: int,
        mul_weight: torch.Tensor,
        eps: float = 1e-5):
    """
    Fuses GroupNorm + Swish + elementwise multiply + Swish into one Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be CUDA"
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    mul_weight = mul_weight.contiguous()

    N, C = x.shape
    assert C % num_groups == 0, "C must be divisible by num_groups"
    group_size = C // num_groups

    out = torch.empty_like(x)
    grid = lambda meta: (N * num_groups,)

    fused_groupnorm_swish_kernel[grid](
        x, gamma, beta, mul_weight, out,
        eps,
        C=C,
        G_SIZE=group_size
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with a fused Triton kernel replacing
    GroupNorm + Swish + element-wise multiply + Swish.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=True)
        self.group_norm = nn.GroupNorm(num_groups, out_features, affine=True)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        # GEMM
        x = self.gemm(x)
        # Fused GroupNorm + Swish + multiply + Swish
        x = fused_groupnorm_swish(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.multiply_weight
        )
        return x


# ----------------------------------------------------------------------------------
# Helper functions (kept identical to original API expectations)
# ----------------------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]


def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]