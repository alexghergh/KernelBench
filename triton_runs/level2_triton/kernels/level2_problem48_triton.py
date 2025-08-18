import torch
import torch.nn as nn
import triton
import triton.language as tl


################################################################################
#                           Triton fused kernel                                #
################################################################################
@triton.jit
def fused_elemwise_kernel(
    x_ptr,           # float*  : convolution output
    scale_ptr,       # float*  : per–out-channel scaling factors (length = C)
    bias_ptr,        # float*  : per–out-channel bias factors    (length = C)
    out_ptr,         # float*  : result tensor
    n_elements,      # int32   : total number of elements in x / out
    channel_size,    # int32   : D * H * W
    channels,        # int32   : C
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load convolution output
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Figure out channel index for every element
    ch_idx = (offsets // channel_size) % channels
    s = tl.load(scale_ptr + ch_idx, mask=mask, other=0.0)
    b = tl.load(bias_ptr + ch_idx, mask=mask, other=0.0)

    # Fused computation:  y = sigmoid( tanh( x * s ) * b )
    x = x * s
    exp2x   = tl.exp(2.0 * x)
    tanh_x  = (exp2x - 1.0) / (exp2x + 1.0)           # tanh
    pre_sig = tanh_x * b
    out     = 1.0 / (1.0 + tl.exp(-pre_sig))          # sigmoid

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_activation(x: torch.Tensor,
                     scaling_factor: torch.Tensor,
                     bias: torch.Tensor) -> torch.Tensor:
    """
    Apply   y = sigmoid( tanh( x * scaling_factor ) * bias )
    using a fused Triton kernel.
    """
    assert x.is_cuda, "Input must reside on GPU"
    x_contig = x.contiguous()
    sf_flat  = scaling_factor.view(-1).contiguous()
    b_flat   = bias.view(-1).contiguous()

    N, C, D, H, W = x_contig.shape
    n_elements    = x_contig.numel()
    channel_size  = D * H * W

    out = torch.empty_like(x_contig)

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_elemwise_kernel[grid](
        x_contig, sf_flat, b_flat, out,
        n_elements, channel_size, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


################################################################################
#                         Autograd-compatible wrapper                          #
################################################################################
class _FusedActivFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scaling_factor, bias):
        output = fused_activation(x, scaling_factor, bias)
        ctx.save_for_backward(x, scaling_factor, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, scaling_factor, bias, output = ctx.saved_tensors

        sf = scaling_factor.view(1, -1, 1, 1, 1)
        bs = bias.view(1, -1, 1, 1, 1)

        x_scaled   = x * sf
        tanh_x     = torch.tanh(x_scaled)
        pre_sig    = tanh_x * bs
        sig_out    = torch.sigmoid(pre_sig)

        grad_p     = grad_output * sig_out * (1 - sig_out)      # dL/dp
        grad_tanh  = grad_p * bs                                # dL/dt
        grad_xsc   = grad_tanh * (1 - tanh_x ** 2)              # dL/da

        grad_input = grad_xsc * sf                              # dL/dx
        grad_sf    = (grad_xsc * x).sum(dim=(0, 2, 3, 4)).view_as(scaling_factor)
        grad_bs    = (grad_p * tanh_x).sum(dim=(0, 2, 3, 4)).view_as(bias)

        return grad_input, grad_sf, grad_bs


def fused_act(x, scaling_factor, bias):
    return _FusedActivFunc.apply(x, scaling_factor, bias)


################################################################################
#                              Optimized Model                                 #
################################################################################
class ModelNew(nn.Module):
    """
    Optimized model: 3D convolution followed by a fused Triton kernel that
    performs scaling, tanh, bias multiplication, and sigmoid in one pass.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 scaling_factor,
                 bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias           = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_act(x, self.scaling_factor, self.bias)
        return x