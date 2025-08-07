# File: scripts/tritonToPtx.py

import os, json, pathlib
from datetime import datetime
import modal

# ----- Image -----
CUDA_VERSION, FLAVOR, OS = "12.4.0", "devel", "ubuntu22.04"
TAG = f"{CUDA_VERSION}-{FLAVOR}-{OS}"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --pre torch triton --index-url https://download.pytorch.org/whl/nightly/cu121",
        "python -m pip install numpy tqdm python-dotenv",
        "echo 'Env ready.'",
    )
    .env({"TMPDIR": "/vol/tmp", "REBUILD_HINT": "v3"})  # bump to break caches
)

# ----- Volume -----
VOLUME_NAME = "triton-kernel-dumps"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ----- App -----
app = modal.App("triton-ptx-dump-v3", image=image, secrets=[])
RUN_OPTS = dict(gpu="A10G", volumes={"/vol": vol})

# ----- Triton kernel (same as yours) -----
import triton, triton.language as tl
import torch, torch.nn as nn
import numpy as np

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_id = pid // (GROUP_SIZE_M * num_pid_n)
    first_pid_m = group_id * GROUP_SIZE_M
    pid_m = first_pid_m + (pid // num_pid_n) % GROUP_SIZE_M
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def _arg_kind(v):
    if isinstance(v, torch.Tensor): return "ptr"
    if isinstance(v, (np.uint32,)): return "u32"
    if isinstance(v, (np.int32,)): return "s32"
    if isinstance(v, (np.float32,)): return "f32"
    if isinstance(v, (np.float64,)): return "f64"
    if isinstance(v, (int,)): return "u32" if 0 <= v <= 0xFFFFFFFF else "u64"
    if isinstance(v, (float,)): return "f32"
    return "unknown"

def _tensor_meta(t: torch.Tensor):
    return {
        "shape":  list(t.shape),
        "stride": list(t.stride()),   # in ELEMENTS
        "dtype":  str(t.dtype),
        "device": str(t.device),
    }

# ----- Compile once and emit PTX + JSON (direct launch, no hooks) -----
@app.function(**RUN_OPTS)
def compile_kernel():
    # tiny tensors to force JIT
    A = torch.empty((8, 8), device="cuda", dtype=torch.float32)
    B = torch.empty((8, 8), device="cuda", dtype=torch.float32)
    C = torch.empty((8, 8), device="cuda", dtype=torch.float32)

    grid = (1,)  # simple 1-block launch; we just want PTX
    launch_kwargs = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_SIZE_M=8)

    matmul_kernel[grid](
        A, B, C,
        8, 8, 8,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        **launch_kwargs,
    )

    # pull compiled artifact from Triton cache
    dev_key  = next(iter(matmul_kernel.cache))
    arg_key  = next(iter(matmul_kernel.cache[dev_key]))
    compiled = matmul_kernel.cache[dev_key][arg_key]
    ptx      = compiled.asm["ptx"]

    num_warps  = getattr(compiled, "num_warps", 4)
    num_stages = getattr(compiled, "num_stages", 2)

    # Build manifest reflecting the exact call above (order matters)
    arg_vals = [
        A, B, C,                     # ptrs
        8, 8, 8,                     # M, N, K
        A.stride(0), A.stride(1),    # stride_am, stride_ak
        B.stride(0), B.stride(1),    # stride_bk, stride_bn
        C.stride(0), C.stride(1),    # stride_cm, stride_cn
    ]
    arg_names = [
        "A", "B", "C",
        "M", "N", "K",
        "stride_am", "stride_ak",
        "stride_bk", "stride_bn",
        "stride_cm", "stride_cn",
    ]

    args_spec, scalars, tensors = [], {}, {}
    for n, v in zip(arg_names, arg_vals):
        k = _arg_kind(v)
        args_spec.append(f"{n}:{k}")
        if k == "ptr":
            tensors[n] = _tensor_meta(v) if isinstance(v, torch.Tensor) else {"note": "non-tensor ptr"}
        else:
            scalars[n] = int(v) if isinstance(v, (int, np.integer)) else float(v)

    dev_name = torch.cuda.get_device_name(0)
    cap_maj, cap_min = torch.cuda.get_device_capability(0)
    sm = cap_maj * 10 + cap_min

    block = [num_warps * 32, 1, 1]
    grid3 = [grid[0], 1, 1]  # normalized to 3D for the JSON

    manifest = {
        "kernel_name": matmul_kernel.fn.__name__,
        "grid": grid3,
        "block": block,
        "shared_mem_bytes": 0,
        "args": args_spec,      # IN CALL ORDER
        "scalars": scalars,
        "tensors": tensors,
        "meta": {
            "num_warps": num_warps,
            "num_stages": num_stages,
            "constexpr": launch_kwargs,
        },
        "device": {"name": dev_name, "sm": sm},
        "notes": [
            "Strides recorded in ELEMENTS (not bytes).",
            "Allocate device buffers for ptr args using 'tensors' metadata.",
            "Bind args in the exact order listed in 'args'.",
            "Launch with grid/block via CUDA Driver API or CuPy.",
        ],
        "ts": datetime.utcnow().isoformat() + "Z",
    }

    out_ptx  = pathlib.Path("/vol/matmul_kernel.ptx")
    out_json = pathlib.Path("/vol/matmul_kernel.json")
    out_ptx.write_text(ptx)
    out_json.write_text(json.dumps(manifest, indent=2))
    print("✅  PTX written to", out_ptx)
    print("✅  JSON written to", out_json)
    vol.commit()

# ----- Entry -----
@app.local_entrypoint()
def main():
    compile_kernel.remote()
