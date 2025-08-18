import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------
#                           Triton Kernels
# --------------------------------------------------------------------
@triton.jit
def maxpool2x2_kernel(
    input_ptr,           # *f32
    output_ptr,          # *f32
    N, C,                # int32
    H_in, W_in,          # int32
    H_out, W_out,        # int32
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    w_o = offsets % W_out
    tmp = offsets // W_out
    h_o = tmp % H_out
    tmp = tmp // H_out
    c = tmp % C
    n = tmp // C

    h_in0 = h_o * 2
    w_in0 = w_o * 2

    base_in = ((n * C + c) * H_in + h_in0) * W_in + w_in0

    idx1 = base_in
    idx2 = base_in + 1
    idx3 = base_in + W_in
    idx4 = base_in + W_in + 1

    v1 = tl.load(input_ptr + idx1, mask=mask, other=-float("inf"))
    v2 = tl.load(input_ptr + idx2, mask=mask, other=-float("inf"))
    v3 = tl.load(input_ptr + idx3, mask=mask, other=-float("inf"))
    v4 = tl.load(input_ptr + idx4, mask=mask, other=-float("inf"))

    out_val = tl.maximum(tl.maximum(v1, v2), tl.maximum(v3, v4))
    tl.store(output_ptr + offsets, out_val, mask=mask)


def triton_maxpool2d_2x2(x: torch.Tensor) -> torch.Tensor:
    """
    2×2 max-pool with stride 2 implemented in Triton.
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    N, C, H_in, W_in = x.shape
    assert H_in % 2 == 0 and W_in % 2 == 0, "Spatial dims must be even."

    H_out, W_out = H_in // 2, W_in // 2
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    maxpool2x2_kernel[grid](
        x,
        out,
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def hardtanh_kernel(
    in_ptr,
    out_ptr,
    min_val,  # f32
    max_val,  # f32
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    x = tl.maximum(x, min_val)
    x = tl.minimum(x, max_val)
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_hardtanh(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Element-wise hardtanh implemented in Triton.
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    hardtanh_kernel[grid](
        x,
        out,
        min_val,
        max_val,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# --------------------------------------------------------------------
#                        Optimized PyTorch Module
# --------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that keeps the native ConvTranspose2d but replaces
    MaxPool2d and Hardtanh with custom Triton kernels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
    ):
        super(ModelNew, self).__init__()

        # Highly-tuned CuDNN kernel – keep as is
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        # Parameters for Triton kernels
        assert (
            maxpool_kernel_size == 2 and maxpool_stride == 2
        ), "Triton max-pool only supports kernel=2, stride=2."
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = triton_maxpool2d_2x2(x)
        x = triton_hardtanh(x, self.hardtanh_min, self.hardtanh_max)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = torch.tanh(x)
        return x


# --------------------------------------------------------------------
#                     Helpers for Benchmark Harness
# --------------------------------------------------------------------
batch_size = 128
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 3
stride = 1
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
    ]