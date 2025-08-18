import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_row_ops_kernel(
    x_ptr,           # pointer to input matrix (B, H)
    out_ptr,         # pointer to output vector (B,)
    scale,           # scalar scale factor
    clamp_min,       # scalar clamp min
    clamp_max,       # scalar clamp max
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)                       # each program handles one row
    row_ptr = x_ptr + row * hidden_size          # pointer to the start of the row

    offs = tl.arange(0, BLOCK_SIZE)              # offsets inside a block

    # ------------------------------------------------------------
    # Pass 1: compute row-wise maximum after scale/residual/clamp
    # ------------------------------------------------------------
    row_max = tl.full([], -float("inf"), dtype=tl.float32)

    for start in range(0, hidden_size, BLOCK_SIZE):
        idx = start + offs
        mask = idx < hidden_size
        x = tl.load(row_ptr + idx, mask=mask, other=-float("inf"))
        x = x * scale          # scale
        x = x + x              # residual addition (x + x)
        x = tl.maximum(tl.minimum(x, clamp_max), clamp_min)  # clamp
        block_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # ------------------------------------------------------------
    # Pass 2: compute log-sum-exp using the row_max for stability
    # ------------------------------------------------------------
    sum_exp = tl.zeros([], dtype=tl.float32)

    for start in range(0, hidden_size, BLOCK_SIZE):
        idx = start + offs
        mask = idx < hidden_size
        x = tl.load(row_ptr + idx, mask=mask, other=0.0)
        x = x * scale
        x = x + x
        x = tl.maximum(tl.minimum(x, clamp_max), clamp_min)
        x = tl.exp(x - row_max)
        sum_exp += tl.sum(x, axis=0)

    lse = row_max + tl.log(sum_exp)              # log-sum-exp
    tl.store(out_ptr + row, lse)                 # store result


def triton_fused_row_ops(x: torch.Tensor, scale: float, clamp_min: float, clamp_max: float) -> torch.Tensor:
    """
    x: Tensor of shape (B, H)
    Returns: Tensor of shape (B, 1) containing logsumexp over dim=1
             after scaling, residual addition, and clamping.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    B, H = x.shape
    out = torch.empty(B, 1, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 1024  # tuned for large rows

    grid = lambda meta: (B,)

    fused_row_ops_kernel[grid](
        x,
        out.squeeze(1),   # pass as 1-D pointer
        scale,
        clamp_min,
        clamp_max,
        hidden_size=H,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model using a Triton kernel to fuse scaling, residual addition,
    clamping, and LogSumExp into one pass.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        x = self.matmul(x)
        x = triton_fused_row_ops(x, self.scale_factor, self.clamp_min, self.clamp_max)
        x = x * F.mish(x)
        return x


batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size, device='cuda')]


def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]