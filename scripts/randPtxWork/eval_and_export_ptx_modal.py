#!/usr/bin/env python
"""eval_and_export_ptx_modal.py
================================
Run a Triton kernel against its reference implementation on a Modal GPU,
using the exact `eval_kernel_against_ref` logic that KernelBench relies on.

In addition to the usual compile/correctness/perf stats, this script gathers
all PTX files produced *during* that evaluation and writes them into a
persistent Modal volume (default: ``triton-ptx-dumps``) **or** a local output
folder when you use it via the local entry-point.

Example (local entry-point, will spin up Modal job under the hood)
----------------------------------------------------------------
python KernelBench/scripts/eval_and_export_ptx_modal.py \
    --ref    KernelBench/level1/1_Square_matrix_multiplication_.py \
    --triton results/triton_runs/my_run/kernels/level1_problem1_triton.py \
    --gpu    A10G \
    --trials 100 \
    --out_dir /vol/ptx_run1

After it finishes check ``/vol/ptx_run1`` in the attached volume or your local
``--out_dir`` for ``*.ptx`` + their companion ``.json`` manifests.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import modal

# -----------------------------------------------------------------------------
# 1.  Modal image & app setup (shares logic with run_triton_generation.py)
# -----------------------------------------------------------------------------
CUDA_VERSION = "12.4.0"
FLAVOR       = "devel"
OS_NAME      = "ubuntu22.04"
TAG          = f"{CUDA_VERSION}-{FLAVOR}-{OS_NAME}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install(
        "torch==2.5.0",
        "triton",  # nightly is fine; version pinned by above torch
        "numpy",
        "datasets",
        "tqdm",
        "python-dotenv",
    )
    # Copy KernelBench and src into the container so "import src" works
    .add_local_dir("ptx-triton-gen/KernelBench", remote_path="/root/KernelBench")
    .add_local_dir("ptx-triton-gen/KernelBench/src", remote_path="/root/src")
    .env({"FORCE_REBUILD_V2": "1"})
)

APP_NAME = "eval-triton-with-ptx-export"
VOLUME_NAME = "triton-ptx-dumps"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image)


# -----------------------------------------------------------------------------
# 2.  Remote class that evaluates + captures PTX
# -----------------------------------------------------------------------------
@app.cls(gpu="A10G", volumes={"/vol": vol})
class TritonEvaluator:
    @modal.method()
    def eval_and_export(
        self,
        ref_src: str,
        triton_src: str,
        gpu_arch: list[str],
        num_perf_trials: int = 100,
        out_dir: str = "/vol",
        verbose: bool = False,
    ) -> dict:
        """Compile & run the Triton kernel, then copy freshly generated PTX files."""
        import os, time, shutil, json, glob, torch

        from src.utils import set_gpu_arch as _set_gpu_arch  # noqa: E402
        from src.eval import eval_kernel_against_ref  # noqa: E402

        _set_gpu_arch(gpu_arch)
        device = torch.device("cuda:0")

        # Timestamp to identify new cache artefacts
        start_ts = time.time()

        # ---- Run evaluation (this compiles & executes the Triton kernel) ----
        result = eval_kernel_against_ref(
            original_model_src=ref_src,
            custom_model_src=triton_src,
            verbose=verbose,
            measure_performance=True,
            num_perf_trials=num_perf_trials,
            backend="triton",
            device=device,
        )

        # ---- Locate PTX files created during this run ----
        cache_root = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
        new_ptx_files: list[str] = []
        for ptx_path in glob.glob(f"{cache_root}/**/*.ptx", recursive=True):
            if os.path.getmtime(ptx_path) >= start_ts - 1:  # small slack
                new_ptx_files.append(ptx_path)

        # ---- Copy to output dir ----
        os.makedirs(out_dir, exist_ok=True)
        saved_paths: list[str] = []
        for src_path in new_ptx_files:
            dst_path = os.path.join(out_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            # also copy the corresponding .json manifest if present
            manifest = os.path.splitext(src_path)[0] + ".json"
            if os.path.exists(manifest):
                shutil.copy2(manifest, os.path.join(out_dir, os.path.basename(manifest)))
            saved_paths.append(dst_path)

        # Persist volume changes when writing to /vol
        if out_dir.startswith("/vol"):
            vol.commit()

        return {
            "kernel_exec_result": result.dict(),
            "ptx_files": saved_paths,
            "out_dir": out_dir,
        }


# -----------------------------------------------------------------------------
# 3.  Local entry-point wrapper
# -----------------------------------------------------------------------------

def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.local_entrypoint()
def main(
    ref: str,
    triton: str,
    gpu: str = "A10G",
    gpu_arch: str | None = None,
    trials: int = 100,
    out_dir: str = "/vol",
    verbose: bool = False,
):
    """Submit the evaluation job to Modal and fetch a summary."""
    ref_src = _read_file(ref)
    triton_src = _read_file(triton)

    arch_list = gpu_arch.split(",") if gpu_arch else ["Ampere"]  # sensible default

    print("🚀  Submitting job to Modal…")
    res = TritonEvaluator.with_options(gpu=gpu)().eval_and_export.remote(
        ref_src,
        triton_src,
        arch_list,
        num_perf_trials=trials,
        out_dir=out_dir,
        verbose=verbose,
    )

    print("\n====== Execution Result ======")
    print("Compiled:", res["kernel_exec_result"]["compiled"])
    print("Correctness:", res["kernel_exec_result"]["correctness"])
    runtime = res["kernel_exec_result"].get("runtime", -1)
    if runtime and runtime > 0:
        print(f"Runtime (us): {runtime:.2f}")

    if res["ptx_files"]:
        print("\nPTX files saved to:")
        for p in res["ptx_files"]:
            print("  •", p)
    else:
        print("No new PTX files detected — did the kernel launch?")

    print("\nDone.") 