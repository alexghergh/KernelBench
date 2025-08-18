import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_swish_bias_gn_kernel(
    x_ptr,            # *float32
    add_bias_ptr,     # *float32
    gn_weight_ptr,    # *float32 (gamma)
    gn_bias_ptr,      # *float32 (beta)
    out_ptr,          # *float32
    C,                # int32 : number of channels (out_features)
    NUM_GROUPS,       # int32 : number of groups
    EPS,              # float32
    BLOCK_SIZE: tl.constexpr,  # channels per group (must equal C // NUM_GROUPS)
):
    pid = tl.program_id(axis=0)

    group_id = pid % NUM_GROUPS          # which group inside the sample
    batch_id = pid // NUM_GROUPS         # which sample in the batch

    # channel indices handled by this program
    c_offsets = group_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # linearized base pointer for this {sample, group}
    base_ptr = batch_id * C + c_offsets

    # ------------------------------------------------------------------
    # Load input and per-channel parameters
    # ------------------------------------------------------------------
    x = tl.load(x_ptr + base_ptr)                       # (BLOCK_SIZE,)
    add_bias = tl.load(add_bias_ptr + c_offsets)        # (BLOCK_SIZE,)
    gamma = tl.load(gn_weight_ptr + c_offsets)          # (BLOCK_SIZE,)
    beta = tl.load(gn_bias_ptr + c_offsets)             # (BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Swish activation and extra bias add: y = swish(x) + bias
    # swish(x) = x * sigmoid(x)
    # ------------------------------------------------------------------
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid + add_bias

    # ------------------------------------------------------------------
    # Compute mean and variance over the channels of this group
    # ------------------------------------------------------------------
    mean = tl.sum(y, axis=0) / BLOCK_SIZE
    var = tl.sum((y - mean) * (y - mean), axis=0) / BLOCK_SIZE
    inv_std = tl.rsqrt(var + EPS)

    # ------------------------------------------------------------------
    # GroupNorm: (y - mean) * inv_std * gamma + beta
    # ------------------------------------------------------------------
    out = (y - mean) * inv_std
    out = out * gamma + beta

    # ------------------------------------------------------------------
    # Store the results
    # ------------------------------------------------------------------
    tl.store(out_ptr + base_ptr, out)


def fused_swish_bias_groupnorm(
    x: torch.Tensor,
    add_bias: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
):
    """
    Fuses Swish activation, per-channel bias addition, and GroupNorm.

    Args:
        x (torch.Tensor):            Input of shape (N, C), CUDA and contiguous.
        add_bias (torch.Tensor):     Per-channel bias added before normalization.  Shape (C,)
        gn_weight (torch.Tensor):    GroupNorm weight (gamma).                   Shape (C,)
        gn_bias (torch.Tensor):      GroupNorm bias  (beta).                    Shape (C,)
        num_groups (int):            Number of groups for GroupNorm.
        eps (float, optional):       Numerical stability term for GroupNorm.
    Returns:
        torch.Tensor: Tensor with same shape as `x`.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dim() == 2, "Expected 2-D tensor (N, C)"
    N, C = x.shape
    assert C % num_groups == 0, "C must be divisible by num_groups"
    group_size = C // num_groups

    # Make sure all tensors are contiguous on device
    x = x.contiguous()
    add_bias = add_bias.contiguous()
    gn_weight = gn_weight.contiguous()
    gn_bias = gn_bias.contiguous()

    out = torch.empty_like(x)

    grid = lambda meta: (N * num_groups,)

    fused_swish_bias_gn_kernel[grid](
        x,
        add_bias,
        gn_weight,
        gn_bias,
        out,
        C,
        num_groups,
        eps,
        BLOCK_SIZE=group_size,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the original model with a custom Triton kernel that
    fuses Swish activation, bias addition, and GroupNorm.
    """

    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)  # Matrix multiplication
        x = fused_swish_bias_groupnorm(
            x,
            self.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
        return x