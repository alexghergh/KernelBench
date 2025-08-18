import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_bn_bias_div_swish_kernel(
    x_ptr,                  # [N, C] input
    mean_ptr,               # [C] running mean
    var_ptr,                # [C] running var
    weight_ptr,             # [C] gamma
    bias_ptr,               # [C] fused beta (bn.bias + extra bias)
    out_ptr,                # [N, C] output
    N,                      # batch size
    C,                      # feature size
    divide_value,           # scalar
    eps,                    # scalar
    BLOCK_SIZE: tl.constexpr  # number of features processed by each program
):
    n_id = tl.program_id(0)            # batch index
    c_block = tl.program_id(1)         # feature block index

    offsets = c_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < C

    x_offsets = n_id * C + offsets

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
    var = tl.load(var_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    inv_std = tl.rsqrt(var + eps)
    y = (x - mean) * inv_std
    y = y * weight + bias
    y = y / divide_value

    sig = 1.0 / (1.0 + tl.exp(-y))
    out = y * sig

    tl.store(out_ptr + x_offsets, out, mask=mask)


def triton_fused_bn_bias_div_swish(
    x: torch.Tensor,                 # [N, C]
    running_mean: torch.Tensor,      # [C]
    running_var: torch.Tensor,       # [C]
    weight: torch.Tensor,            # [C]
    fused_bias: torch.Tensor,        # [C]
    divide_value: float,
    eps: float,
    block_size: int = 128,
):
    """
    Fuses BatchNorm (inference), additional bias addition, scalar division,
    and Swish activation into a single Triton kernel.
    """
    assert x.is_cuda, "Input must be on CUDA"

    N, C = x.shape
    x_contig = x.contiguous()
    mean_contig = running_mean.contiguous()
    var_contig = running_var.contiguous()
    weight_contig = weight.contiguous()
    bias_contig = fused_bias.contiguous()

    out = torch.empty_like(x_contig)

    grid = (N, (C + block_size - 1) // block_size)

    fused_bn_bias_div_swish_kernel[grid](
        x_contig,
        mean_contig,
        var_contig,
        weight_contig,
        bias_contig,
        out,
        N,
        C,
        float(divide_value),
        float(eps),
        BLOCK_SIZE=block_size,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for the fused sequence:
    BatchNorm (inference) + bias addition + division + Swish activation.
    Falls back to the standard PyTorch path during training.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bn_eps=1e-5,
        bn_momentum=0.1,
        bias_shape=(1,),
        divide_value=1.0,
    ):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, x: torch.Tensor):
        x = self.matmul(x)

        # Use Triton kernel for the inference path
        if not self.training:
            fused_bias = self.bn.bias + self.bias  # broadcasting handled by PyTorch
            x = triton_fused_bn_bias_div_swish(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                fused_bias,
                self.divide_value,
                self.bn.eps,
            )
            return x

        # Fallback to standard PyTorch ops during training
        x = self.bn(x)
        x = x + self.bias
        x = x / self.divide_value
        x = x * torch.sigmoid(x)
        return x