import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr,                                       # input tensor
    out_ptr,                                     # output tensor
    N, C, D, H, W,                               # input dimensions
    OD, OH, OW,                                  # output dimensions
    SD, SH, SW,                                  # stride (d, h, w)
    PD, PH, PW,                                  # padding (d, h, w)
    DD, DH, DW,                                  # dilation (d, h, w)
    in_strideN, in_strideC, in_strideD, in_strideH, in_strideW,
    out_strideN, out_strideC, out_strideD, out_strideH, out_strideW,
    BLOCK_SIZE: tl.constexpr,                    # one element per program
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr  # kernel sizes
):
    pid = tl.program_id(0)                       # unique program ID
    idx = pid

    w_out = idx % OW
    idx //= OW
    h_out = idx % OH
    idx //= OH
    d_out = idx % OD
    idx //= OD
    c = idx % C
    idx //= C
    n = idx

    d_in_start = d_out * SD - PD
    h_in_start = h_out * SH - PH
    w_in_start = w_out * SW - PW

    max_val = tl.full((), -1e30, dtype=tl.float32)  # initialize with very small number

    for kd in range(KD):
        id = d_in_start + kd * DD
        valid_d = (id >= 0) & (id < D)
        id_safe = tl.where(valid_d, id, 0)

        for kh in range(KH):
            ih = h_in_start + kh * DH
            valid_h = (ih >= 0) & (ih < H)
            ih_safe = tl.where(valid_h, ih, 0)

            for kw in range(KW):
                iw = w_in_start + kw * DW
                valid_w = (iw >= 0) & (iw < W)
                iw_safe = tl.where(valid_w, iw, 0)

                mask = valid_d & valid_h & valid_w

                offset = (
                    n * in_strideN
                    + c * in_strideC
                    + id_safe * in_strideD
                    + ih_safe * in_strideH
                    + iw_safe * in_strideW
                )

                val = tl.load(x_ptr + offset, mask=mask, other=-1e30)
                max_val = tl.maximum(max_val, val)

    out_offset = (
        n * out_strideN
        + c * out_strideC
        + d_out * out_strideD
        + h_out * out_strideH
        + w_out * out_strideW
    )
    tl.store(out_ptr + out_offset, max_val)


def triton_maxpool3d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int | None = None,
    padding: int = 0,
    dilation: int = 1,
):
    """
    Triton implementation of 3D Max Pooling for tensors in NCDHW format.
    Only supports:
      • identical kernel sizes in all three dimensions
      • identical stride/padding/dilation in all three dimensions
      • ceil_mode=False, return_indices=False
    """

    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.ndim == 5, "Input must be of shape (N, C, D, H, W)."

    # Normalize hyper-parameters
    stride = stride if stride is not None else kernel_size
    KD = KH = KW = kernel_size
    SD = SH = SW = stride
    PD = PH = PW = padding
    DD = DH = DW = dilation

    N, C, D, H, W = x.shape
    OD = (D + 2 * PD - DD * (KD - 1) - 1) // SD + 1
    OH = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1

    x_contig = x.contiguous()
    out = torch.empty((N, C, OD, OH, OW), dtype=x.dtype, device=x.device)

    # Strides for contiguous NCDHW tensor
    in_strideW = 1
    in_strideH = W
    in_strideD = H * W
    in_strideC = D * H * W
    in_strideN = C * D * H * W

    out_strideW = 1
    out_strideH = OW
    out_strideD = OH * OW
    out_strideC = OD * OH * OW
    out_strideN = C * OD * OH * OW

    num_progs = N * C * OD * OH * OW
    grid = lambda meta: (num_progs,)

    maxpool3d_kernel[grid](
        x_contig,
        out,
        N,
        C,
        D,
        H,
        W,
        OD,
        OH,
        OW,
        SD,
        SH,
        SW,
        PD,
        PH,
        PW,
        DD,
        DH,
        DW,
        in_strideN,
        in_strideC,
        in_strideD,
        in_strideH,
        in_strideW,
        out_strideN,
        out_strideC,
        out_strideD,
        out_strideH,
        out_strideW,
        BLOCK_SIZE=1,
        KD=KD,
        KH=KH,
        KW=KW,
    )
    return out


class ModelNew(nn.Module):
    """
    Model that applies 3-D Max Pooling using a custom Triton kernel.
    Only supports ceil_mode=False and return_indices=False.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        assert not return_indices, "return_indices=True is not supported."
        assert not ceil_mode, "ceil_mode=True is not supported."
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


# ---------------------------------------------------------------------
# Helper functions required by the evaluation framework
# ---------------------------------------------------------------------
batch_size = 16
channels = 32
dim1 = 128
dim2 = 128
dim3 = 128
kernel_size = 3
stride = 2
padding = 1
dilation = 3


def get_inputs():
    x = torch.rand(batch_size, channels, dim1, dim2, dim3, device="cuda")
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]