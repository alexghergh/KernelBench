import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr,            # *Pointer* to the input tensor
    out_ptr,          # *Pointer* to the output tensor
    L_in,             # Input length  (int32)
    L_out,            # Output length (int32)
    stride,           # Stride        (int32)
    padding,          # Padding       (int32)
    BLOCK_N: tl.constexpr,      # Number of output elements computed by each program
    KERNEL_SIZE: tl.constexpr,  # Pooling window size (compile-time constant)
):
    # 2-D launch grid:
    #  - program_id(0): different (batch, channel) rows
    #  - program_id(1): different blocks along the length dimension
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)

    # Compute base pointers for this row
    x_row_ptr = x_ptr + row_id * L_in
    out_row_ptr = out_ptr + row_id * L_out

    # Indices of the BLOCK_N output elements handled by this program
    out_idx = col_id * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = out_idx < L_out

    # Starting index (in the input) of every pooling window
    in_start = out_idx * stride - padding

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Accumulate over the pooling window
    for k in range(KERNEL_SIZE):
        idx = in_start + k
        valid = (idx >= 0) & (idx < L_in)
        mask = out_mask & valid
        idx_clamped = tl.where(valid, idx, 0)           # clamp to avoid OOB pointer math
        val = tl.load(x_row_ptr + idx_clamped, mask=mask, other=0.0)
        acc += val

    acc = acc / KERNEL_SIZE
    tl.store(out_row_ptr + out_idx, acc, mask=out_mask)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int = 1, padding: int = 0):
    """
    Fast AvgPool1d implemented with Triton.

    Supports float32 CUDA tensors that are contiguous in memory and laid out as (B, C, L).
    Fallback to torch.nn.functional.avg_pool1d for other cases is provided in ModelNew.
    """
    assert x.is_cuda, "Input tensor must be on CUDA to use the Triton kernel."
    assert x.dtype == torch.float32, "Only float32 tensors are supported by the Triton kernel."
    x = x.contiguous()

    B, C, L_in = x.shape
    L_out = (L_in + 2 * padding - kernel_size) // stride + 1
    out = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)

    BLOCK_N = 128
    grid = (B * C, triton.cdiv(L_out, BLOCK_N))

    avg_pool1d_kernel[grid](
        x, out,
        L_in, L_out,
        stride, padding,
        BLOCK_N=BLOCK_N,
        KERNEL_SIZE=kernel_size,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D Average Pooling using a custom Triton kernel.
    Falls back to PyTorch's implementation for non-CUDA or non-float32 inputs.
    """

    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32:
            return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)
        # Fallback for CPU tensors or unsupported dtypes
        return F.avg_pool1d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)


# ----------------------------------
# Helper functions (unchanged logic)
# ----------------------------------
batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4


def get_inputs():
    # Move the tensor to GPU for the Triton kernel
    x = torch.rand(batch_size, in_channels, input_length).cuda()
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding]