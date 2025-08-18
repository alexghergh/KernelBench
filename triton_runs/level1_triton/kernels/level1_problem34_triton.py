import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr,               # *float32
    out_ptr,             # *float32
    hw,                  # int  = H * W
    eps,                 # float32
    BLOCK_SIZE: tl.constexpr,  # number of elements handled per program step
):
    pid = tl.program_id(0)                # one program per (N, C) pair
    row_start = pid * hw                  # starting offset of this row

    # ------------------------------------------------------------------
    # Pass 1: compute mean and variance for this (N, C) feature map
    # ------------------------------------------------------------------
    mean_acc = tl.zeros((), dtype=tl.float32)
    var_acc = tl.zeros((), dtype=tl.float32)

    offset = 0
    while offset < hw:
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < hw
        ptrs = x_ptr + row_start + offs
        x = tl.load(ptrs, mask=mask, other=0.0)

        mean_acc += tl.sum(x, axis=0)
        var_acc += tl.sum(x * x, axis=0)

        offset += BLOCK_SIZE

    mean = mean_acc / hw
    var = var_acc / hw - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # ------------------------------------------------------------------
    # Pass 2: normalize and store
    # ------------------------------------------------------------------
    offset = 0
    while offset < hw:
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < hw
        ptrs = x_ptr + row_start + offs
        x = tl.load(ptrs, mask=mask, other=0.0)

        y = (x - mean) * inv_std
        tl.store(out_ptr + row_start + offs, y, mask=mask)

        offset += BLOCK_SIZE


def triton_instance_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Instance-Norm implementation backed by a Triton kernel.
    Supports NCHW layout, affine=False, track_running_stats=False.
    """
    assert x.is_cuda and x.ndim == 4, "Input must be a 4-D CUDA tensor (N, C, H, W)."

    N, C, H, W = x.shape
    hw = H * W
    total_rows = N * C

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    BLOCK_SIZE = 1024
    grid = lambda meta: (total_rows,)

    instance_norm_kernel[grid](
        x_contig,
        out,
        hw,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class TritonInstanceNorm2d(nn.Module):
    """
    Drop-in replacement for nn.InstanceNorm2d using the above Triton kernel.
    Only supports the default settings (affine=False, track_running_stats=False).
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cuda":
            return triton_instance_norm(x, eps=self.eps)
        # Fallback to PyTorch on CPU tensors
        return torch.nn.functional.instance_norm(
            x,
            running_mean=None,
            running_var=None,
            weight=None,
            bias=None,
            eps=self.eps,
        )


class ModelNew(nn.Module):
    """
    Optimized model with Triton-based Instance Normalization.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.inorm = TritonInstanceNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)


# ----------------------------------------------------------------------
# I/O helpers (match original API)
# ----------------------------------------------------------------------
batch_size = 112
features = 64
dim1 = 512
dim2 = 512


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2, device="cuda")
    return [x]


def get_init_inputs():
    return [features]