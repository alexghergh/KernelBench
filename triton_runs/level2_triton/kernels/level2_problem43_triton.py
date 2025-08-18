import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _logsumexp_relu_kernel(
    x_ptr,          # pointer to input rows [rows, C]
    out_ptr,        # pointer to output [rows]
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)                      # each program handles one (N*D*H*W) position
    offs   = tl.arange(0, BLOCK_SIZE)              # channel indices handled by this program
    mask   = offs < C                              # mask out-of-range elements

    ptrs = x_ptr + row_id * C + offs               # pointers to the channels we need
    neg_inf = tl.full((), -1.0e30, tl.float32)     # a safe -inf substitute

    x = tl.load(ptrs, mask=mask, other=neg_inf)    # load channels (masked)

    m   = tl.max(x, axis=0)                        # max over channels
    s   = tl.sum(tl.exp(x - m), axis=0)            # sumexp(x - m)
    lse = m + tl.log(s)                            # logsumexp
    lse = tl.maximum(lse, 0.0)                     # ReLU

    tl.store(out_ptr + row_id, lse)                # write result


def triton_logsumexp_relu(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Fused logsumexp (along `dim`) + ReLU implemented with Triton.
    Only supports `dim == 1` (channel dimension) for 5-D tensors [N, C, D, H, W].
    """
    assert x.is_cuda,           "Input must be on CUDA"
    assert x.dim() == 5,        "Only 5-D tensors supported"
    assert dim == 1,            "Only channel dimension reduction supported"

    N, C, D, H, W = x.shape

    # Move channel to the innermost dim so every row is contiguous in memory
    x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)   # [rows, C]
    rows   = x_flat.shape[0]

    out = torch.empty(rows, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = triton.next_power_of_2(C)                       # one thread per channel
    grid = (rows,)

    _logsumexp_relu_kernel[grid](x_flat, out, C, BLOCK_SIZE)

    out = out.view(N, D, H, W).unsqueeze(1).contiguous()         # back to [N, 1, D, H, W]
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses logsumexp and ReLU with a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv      = nn.Conv3d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding)
        self.max_pool  = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = triton_logsumexp_relu(x, dim=1)
        return x


# -------------------------------------------------
# Helper functions required by the autograder
# -------------------------------------------------
def get_inputs():
    batch_size = 4
    in_channels = 32
    depth, height, width = 32, 128, 128
    return [torch.rand(batch_size, in_channels, depth, height, width, device="cuda")]


def get_init_inputs():
    in_channels  = 32
    out_channels = 64
    kernel_size  = 3
    stride       = 1
    padding      = 1
    return [in_channels, out_channels, kernel_size, stride, padding]