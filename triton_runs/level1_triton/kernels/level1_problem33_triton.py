import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bn_inference_kernel(
    x_ptr,         # *T
    out_ptr,       # *T
    scale_ptr,     # *T
    bias_ptr,      # *T
    NHW,           # int32  = N*H*W
    n_elements,    # int32  = total elements (N*C*H*W)
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)       # Channel index
    pid_blk = tl.program_id(1)     # Block along (N, H, W)

    offsets = pid_blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx = pid_c * NHW + offsets       # Linear indices this program handles
    mask = idx < n_elements           # Guard against OOB

    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)
    y = x * scale + bias
    tl.store(out_ptr + idx, y, mask=mask)


def triton_batch_norm_2d_inference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    """
    Fast BatchNorm2d inference using Triton.
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernels."

    N, C, H, W = x.shape
    NHW = N * H * W
    n_elements = x.numel()

    # Pre-compute per-channel scale and bias
    scale = weight / torch.sqrt(running_var + eps)
    bias_corr = bias - running_mean * scale

    x = x.contiguous()
    scale = scale.to(dtype=x.dtype, device=x.device).contiguous()
    bias_corr = bias_corr.to(dtype=x.dtype, device=x.device).contiguous()
    out = torch.empty_like(x)

    grid = (C, triton.cdiv(NHW, BLOCK_SIZE))
    bn_inference_kernel[grid](
        x, out, scale, bias_corr,
        NHW, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses a Triton kernel for BatchNorm2d during inference,
    falling back to PyTorch's implementation during training or on CPU tensors.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda) or self.training:
            # Training / CPU path: use standard PyTorch BatchNorm
            return self.bn(x)
        else:
            # Inference on CUDA: use Triton-accelerated path
            return triton_batch_norm_2d_inference(
                x,
                self.bn.weight.detach(),
                self.bn.bias.detach(),
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
            )


# ----------------------------------------------------------------------------------
# Utility functions mirroring the original interface (useful for external harnesses)
# ----------------------------------------------------------------------------------
batch_size = 64
features = 64
dim1 = 512
dim2 = 512


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]


def get_init_inputs():
    return [features]