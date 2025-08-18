import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_mul_kernel(
    x_ptr,        # Pointer to input tensor (float32)
    mask_ptr,     # Pointer to boolean mask (int1)
    out_ptr,      # Pointer to output tensor (float32)
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_valid = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=is_valid, other=0.0)
    m = tl.load(mask_ptr + offsets, mask=is_valid, other=0)
    m = m.to(tl.float32)

    tl.store(out_ptr + offsets, x * m, mask=is_valid)


def triton_masked_mul(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and mask.is_cuda, "Inputs must be CUDA tensors."
    assert x.shape == mask.shape, "x and mask must have the same shape."

    x = x.contiguous()
    mask = mask.contiguous()

    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    masked_mul_kernel[grid](x, mask, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        if not x.is_cuda:
            x = x.cuda()
        if not mask.is_cuda:
            mask = mask.cuda()

        masked = triton_masked_mul(x, mask)
        return torch.cumsum(masked, dim=self.dim)


batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    x = torch.rand(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()
    return [x, mask]


def get_init_inputs():
    return [dim]