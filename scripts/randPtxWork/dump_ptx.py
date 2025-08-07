import os
import pathlib
import modal

# ----------------------------------------------------------------------
# 0)  Choose container base & build steps – matches your previous style
# ----------------------------------------------------------------------
CUDA_VERSION   = "12.4.0"
FLAVOR         = "devel"
OS             = "ubuntu22.04"
TAG            = f"{CUDA_VERSION}-{FLAVOR}-{OS}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    # --- MODIFIED PART ---
    .run_commands(
        "python -m pip install --upgrade pip",
        # Use the cu121 nightly index, which is very reliable for torch/triton compatibility.
        # This will install the latest PyTorch and Triton nightlies built against the CUDA 12.1 toolkit.
        # Your CUDA 12.4 container can run this without issue.
        "python -m pip install --pre torch triton --index-url https://download.pytorch.org/whl/nightly/cu121",
        # Install other packages separately
        "python -m pip install numpy tqdm python-dotenv",
    )
    .env(
        {
            "FORCE_REBUILD_V2": "1",
            "TMPDIR": "/vol/tmp", # torch-inductor & Triton caches land here
        }
    )
)
# ----------------------------------------------------------------------
# 1)  Persisted volume – keeps artifacts between runs exactly like before
# ----------------------------------------------------------------------
VOLUME_NAME = "triton-kernel-dumps"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ----------------------------------------------------------------------
# 2)  Modal stub
# ----------------------------------------------------------------------
app = modal.App("triton-ptx-dump", image=image, secrets=[])

# Mount the volume at /vol inside the container
RUN_OPTS = dict(gpu="A10G", volumes={"/vol": vol})

# ----------------------------------------------------------------------
# 3)  Triton kernel -- *unchanged* from your original question
# ----------------------------------------------------------------------
import triton
import triton.language as tl
import torch
import torch.nn as nn


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,                # pointers to matrices
    M, N, K,                            # dimensions
    stride_am, stride_ak,               # strides for A
    stride_bk, stride_bn,               # strides for B
    stride_cm, stride_cn,               # strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ------------------------
    # Program ID calculations
    # ------------------------
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_id = pid // (GROUP_SIZE_M * num_pid_n)
    first_pid_m = group_id * GROUP_SIZE_M
    pid_m = first_pid_m + (pid // num_pid_n) % GROUP_SIZE_M
    pid_n = pid % num_pid_n

    # ------------------------
    # Create index ranges for the tile
    # ------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # pointers for A and B
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------------------
    # Loop over K dimension
    # ------------------------
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # ------------------------
    # Write back the result
    # ------------------------
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        C_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication C = A @ B using a custom Triton kernel.
    Supports float32 inputs stored in row-major (contiguous) layout.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 supported."

    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Inner dimensions must agree."

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        GROUP_SIZE_M=8,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs square matrix multiplication using a Triton kernel.
    """
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


# Input size (square matrices)
N = 2048 * 2


def get_inputs():
    A = torch.rand(N, N, device="cuda", dtype=torch.float32)
    B = torch.rand(N, N, device="cuda", dtype=torch.float32)
    return [A, B]


def get_init_inputs():
    return []

# ----------------------------------------------------------------------
# 4)  Function that compiles once and writes matmul_kernel.ptx into /vol
# ----------------------------------------------------------------------
@app.function(**RUN_OPTS)
def compile_kernel():
    # Tiny dummy tensors to trigger the JIT
    A = torch.empty((8, 8), device="cuda", dtype=torch.float32)
    B = torch.empty((8, 8), device="cuda", dtype=torch.float32)

    matmul_kernel[(1,)](
        A, B, A,
        8, 8, 8,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        A.stride(0), A.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_SIZE_M=8,
    )

    # Pull PTX from Triton’s cache
    dev_key  = next(iter(matmul_kernel.cache))
    arg_key  = next(iter(matmul_kernel.cache[dev_key]))
    ptx      = matmul_kernel.cache[dev_key][arg_key].asm["ptx"]

    out_path = pathlib.Path("/vol/matmul_kernel.ptx")
    out_path.write_text(ptx)
    print("✅  PTX written to", out_path)

    # Persist changes
    vol.commit()


# ----------------------------------------------------------------------
# 5)  Local entrypoint – kicks off the remote GPU job
# ----------------------------------------------------------------------
@app.local_entrypoint()
def main():
    compile_kernel.remote()