import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_softmax_sigmoid_kernel(
    x_ptr,        # Pointer to the (flattened) input tensor
    out_ptr,      # Pointer to the (flattened) output tensor
    C,            # Number of channels (softmax length)
    BLOCK_SIZE: tl.constexpr,  # Number of elements processed by a single program
):
    pid = tl.program_id(0)               # Row index this program will process
    row_start = pid * C                  # Offset of the first element of the row

    offsets = tl.arange(0, BLOCK_SIZE)   # [0, ..., BLOCK_SIZE-1]
    mask = offsets < C                   # Mask to stay within row bounds

    # Load data for this row
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-1e30)

    # Softmax: subtract max for numerical stability
    row_max = tl.max(x, 0)
    x = tl.exp(x - row_max)

    # Compute denominator (sum of exps) and normalize
    denom = tl.sum(x, 0)
    softmax = x / denom

    # Apply sigmoid to the softmax result
    sigmoid = 1.0 / (1.0 + tl.exp(-softmax))

    # Store results
    tl.store(out_ptr + row_start + offsets, sigmoid, mask=mask)


def fused_softmax_sigmoid(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Fused Softmax (along channel dimension) followed by Sigmoid,
    implemented with a custom Triton kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, D, H, W) on CUDA.
        block_size (int, optional): Number of elements processed per kernel instance.

    Returns:
        torch.Tensor: Output tensor with the same shape as `x`.
    """
    assert x.is_cuda, "Input tensor must reside on GPU."
    N, C, D, H, W = x.shape

    # Make channel dimension contiguous (move C to the last axis)
    x_perm = x.permute(0, 2, 3, 4, 1).contiguous()  # (N, D, H, W, C)
    x_flat = x_perm.view(-1, C)                     # (N*D*H*W, C)

    out_flat = torch.empty_like(x_flat)

    # Launch one kernel instance per "row" (i.e., per spatial position)
    grid = lambda meta: (x_flat.shape[0],)
    fused_softmax_sigmoid_kernel[grid](
        x_flat, out_flat,
        C,
        BLOCK_SIZE=block_size,
    )

    # Restore original layout
    out_perm = out_flat.view(N, D, H, W, C)
    out = out_perm.permute(0, 4, 1, 2, 3).contiguous()
    return out


class ModelNew(nn.Module):
    """
    Optimized model where Softmax and Sigmoid are fused into a single
    custom Triton kernel for better performance.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias=True,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_softmax_sigmoid(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions required by the evaluation harness
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W, device="cuda")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]