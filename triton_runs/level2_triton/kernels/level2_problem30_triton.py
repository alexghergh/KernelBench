import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_hardtanh_kernel(
    x_ptr,          # float32 *, shape [B, C]
    w_ptr,          # float32 *, shape [C]
    b_ptr,          # float32 *, shape [C]
    o_ptr,          # float32 *, shape [B, C]
    stride_features,  # int32  : distance between consecutive rows (== C)
    eps,              # float32
    hard_min,         # float32
    hard_max,         # float32
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Identify the (batch_idx, group_idx) pair this program will process
    pid = tl.program_id(0)
    batch_idx = pid // NUM_GROUPS
    group_idx = pid % NUM_GROUPS

    # Channels handled by this program
    offs = tl.arange(0, BLOCK_SIZE)
    ch   = group_idx * GROUP_SIZE + offs
    mask = offs < GROUP_SIZE

    # Pointers
    x_ptrs = x_ptr + batch_idx * stride_features + ch
    w_ptrs = w_ptr + ch
    b_ptrs = b_ptr + ch
    o_ptrs = o_ptr + batch_idx * stride_features + ch

    # Load input and affine params
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    w = tl.load(w_ptrs, mask=mask, other=1.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    # Compute mean & variance across the channels of this group
    mean = tl.sum(x, axis=0) / GROUP_SIZE
    diff = x - mean
    var  = tl.sum(diff * diff, axis=0) / GROUP_SIZE
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize, apply affine transform, then HardTanh clamp
    y = diff * inv_std
    y = y * w + b
    y = tl.minimum(tl.maximum(y, hard_min), hard_max)

    # Store result
    tl.store(o_ptrs, y, mask=mask)


def fused_group_norm_hardtanh(x: torch.Tensor,
                              weight: torch.Tensor,
                              bias: torch.Tensor,
                              num_groups: int,
                              hard_min: float,
                              hard_max: float,
                              eps: float = 1e-5) -> torch.Tensor:
    """
    Fused GroupNorm(+affine) + HardTanh implemented with Triton.
    Falls back to PyTorch when tensors are on CPU.
    """
    if not x.is_cuda:
        y = torch.nn.functional.group_norm(x, num_groups, weight, bias, eps=eps)
        return torch.clamp(y, min=hard_min, max=hard_max)

    B, C = x.shape
    assert C % num_groups == 0, "C must be divisible by num_groups"
    group_size = C // num_groups

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = group_size  # 512 for the provided config
    grid = lambda meta: (B * num_groups,)

    group_norm_hardtanh_kernel[grid](
        x, weight, bias, out,
        C,
        eps,
        hard_min,
        hard_max,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the original model.
    Uses a Triton kernel that fuses GroupNorm and HardTanh for better performance.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_groups: int,
                 hardtanh_min: float,
                 hardtanh_max: float,
                 eps: float = 1e-5):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.num_groups = num_groups
        self.eps = eps
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # GroupNorm learnable parameters
        self.weight = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)
        x = fused_group_norm_hardtanh(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.hardtanh_min,
            self.hardtanh_max,
            self.eps,
        )
        return x