import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduce_dim1_kernel(
    x_ptr,           # *fp32,   [B, D1, D2]
    out_ptr,         # *fp32,   [B, 1, D2]
    D1,              # reduction size
    D2,              # inner dimension
    stride_batch,    # = D1 * D2
    stride_d1,       # = D2
    BLOCK_K: tl.constexpr,  # number of reduction elements processed per iteration
):
    pid = tl.program_id(0)          # unique program id
    col = pid % D2                  # index in dim2
    batch = pid // D2               # batch index

    # pointer to (batch, 0, col)
    base_ptr = x_ptr + batch * stride_batch + col

    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros([], dtype=tl.float32)

    for k in range(0, D1, BLOCK_K):
        k_offsets = k + offs_k
        mask = k_offsets < D1
        ptrs = base_ptr + k_offsets * stride_d1
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    out_ptr += batch * D2 + col
    tl.store(out_ptr, acc)


def triton_sum_dim1(x: torch.Tensor, block_k: int = 1024) -> torch.Tensor:
    """
    Sum-reduce `x` along dimension 1 using a custom Triton kernel.
    Keeps the reduced dimension (keepdim=True).
    """
    if not x.is_cuda:
        x = x.cuda()
    x = x.contiguous()
    assert x.dtype == torch.float32, "Only fp32 is supported in this kernel."
    assert x.dim() == 3, "Input must be a 3-D tensor (B, D1, D2)."

    B, D1, D2 = x.shape
    out = torch.empty(B, 1, D2, device=x.device, dtype=x.dtype)

    stride_batch = D1 * D2
    stride_d1 = D2

    grid = lambda meta: (B * D2,)

    sum_reduce_dim1_kernel[grid](
        x,
        out,
        D1,
        D2,
        stride_batch,
        stride_d1,
        BLOCK_K=block_k,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over dimension 1
    using a custom Triton kernel.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        assert dim == 1, "ModelNew currently supports reduction over dim==1 only."
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_dim1(x)