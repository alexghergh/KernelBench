import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _constant_fill_kernel(
    out_ptr,          # pointer to the output tensor
    value,            # scalar value to write
    n_elements,       # total number of elements to write
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, value, mask=mask)


def triton_constant_fill(out_tensor: torch.Tensor, value: float):
    """
    Fills `out_tensor` with the scalar `value` using a Triton kernel.
    """
    assert out_tensor.is_cuda, "Tensor must reside on CUDA device."
    out_tensor = out_tensor.contiguous()
    n_elements = out_tensor.numel()
    BLOCK_SIZE = 1024

    # Grid: one block per chunk of BLOCK_SIZE elements
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _constant_fill_kernel[grid](out_tensor, value, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out_tensor


class ModelNew(nn.Module):
    """
    Optimized model that replaces the original Conv3d → GroupNorm →
    min → clamp chain with a single Triton kernel that produces the
    constant tensor `min_value`, followed by dropout.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        min_value,
        max_value,
        dropout_p,
    ):
        super().__init__()
        # Preserve parameters needed to compute output shape
        self.out_channels = out_channels
        self.dropout_p = dropout_p
        self.min_value = float(min_value)

        # Normalize kernel size to a triple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

    def _output_spatial_shape(self, D, H, W):
        kd, kh, kw = self.kernel_size
        D_out = D - kd + 1
        H_out = H - kh + 1
        W_out = W - kw + 1
        return D_out, H_out, W_out

    def forward(self, x):
        """
        The original computation chain always collapsed every element
        to `min_value` before dropout.  We exploit this to skip the
        expensive Conv3d and GroupNorm, directly generating the
        constant tensor with a Triton kernel and then applying dropout.
        """
        N, _, D, H, W = x.shape
        D_out, H_out, W_out = self._output_spatial_shape(D, H, W)

        out = torch.empty(
            (N, self.out_channels, D_out, H_out, W_out),
            dtype=x.dtype,
            device=x.device,
        )

        # Fill the tensor with the constant `min_value` via Triton
        triton_constant_fill(out, self.min_value)

        # Apply dropout (same semantics as nn.Dropout)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        return out