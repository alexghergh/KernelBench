import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_dim1_kernel(
    x_ptr,                   # *float32
    out_ptr,                 # *int32
    stride_b, stride_d1, stride_d2,  # tensor strides
    dim2_size,               # size of dim2  (W)
    dim1_size,               # size along which we reduce (L)
    BLOCK_SIZE: tl.constexpr = 128,
):
    """
    Each program computes the argmin along dim1 (length = dim1_size) for one
    (batch_idx, col_idx) pair – i.e. for one contiguous vector of length dim1_size
    whose address is:
        x_ptr + batch_idx * stride_b + col_idx * stride_d2
    The result is written to:
        out_ptr[batch_idx, col_idx]
    """
    seq_id = tl.program_id(0)
    # Map 1-D program id to (batch, col) coordinates
    batch_id = seq_id // dim2_size
    col_id = seq_id % dim2_size

    # Pointer to first element of the vector this program will process
    base_ptr = x_ptr + batch_id * stride_b + col_id * stride_d2

    # Initial global minimum
    global_min_val = tl.full((), float("inf"), tl.float32)
    global_min_idx = tl.zeros((), tl.int32)

    # Iterate over the reduction dimension in BLOCK_SIZE chunks
    for start in range(0, dim1_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < dim1_size

        vals = tl.load(base_ptr + offs * stride_d1, mask=mask, other=float("inf"))

        # Find minimum value in the current block
        block_min_val = tl.min(vals, axis=0)

        # Indices of elements equal to block_min_val
        idxs = offs
        same_min = vals == block_min_val
        block_min_idx = tl.min(tl.where(same_min, idxs, dim1_size), axis=0)

        # Update global minimum (value + index)
        is_smaller = block_min_val < global_min_val
        is_equal = block_min_val == global_min_val

        global_min_val = tl.where(is_smaller, block_min_val, global_min_val)
        global_min_idx = tl.where(is_smaller, block_min_idx, global_min_idx)
        # Tie-break: keep the smallest index if values are equal
        global_min_idx = tl.where(is_equal & (block_min_idx < global_min_idx),
                                  block_min_idx, global_min_idx)

    # Store result
    tl.store(out_ptr + seq_id, global_min_idx)


def triton_argmin_dim1(x: torch.Tensor) -> torch.Tensor:
    """
    Argmin along dimension 1 implemented with a Triton kernel.
    Only supports 3-D contiguous CUDA tensors and dim == 1 (as used by the model).
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dim() == 3, "Only 3-D tensors are supported"
    x = x.contiguous()

    B, L, W = x.shape  # batch, dim1 (reduction), dim2
    out_int32 = torch.empty((B, W), dtype=torch.int32, device=x.device)

    stride_b, stride_d1, stride_d2 = x.stride()

    grid = (B * W,)

    argmin_dim1_kernel[grid](
        x, out_int32,
        stride_b, stride_d1, stride_d2,
        W, L,
        BLOCK_SIZE=128,
    )

    return out_int32.to(torch.long)


class ModelNew(nn.Module):
    """
    Optimized model that computes argmin along a specified dimension.
    Uses a custom Triton kernel when dim == 1 and the tensor is on CUDA;
    falls back to torch.argmin otherwise.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1 and x.is_cuda:
            return triton_argmin_dim1(x)
        else:
            return torch.argmin(x, dim=self.dim)