#!/usr/bin/env python
"""eval_triton_vs_ref.py
=================================
Given two source files:
  1) a reference PyTorch implementation (used in KernelBench)
  2) a custom Triton implementation that defines a class ``ModelNew`` and matching
     ``get_init_inputs()`` / ``get_inputs()`` helpers (same convention used by
     run_triton_generation.py)

The script calls ``src.eval.eval_kernel_against_ref`` to compile, check
correctness, and (optionally) benchmark the Triton kernel, then pretty-prints
the resulting metrics.

Example
-------
python eval_triton_vs_ref.py \
    --ref   KernelBench/level1/1_Square_matrix_multiplication_.py \
    --triton results/triton_runs/my_run/kernels/level1_problem1_triton.py \
    --device 0 --trials 100 --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import torch

# KernelBench utilities live one directory up from this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Local imports after path tweak
from src.eval import eval_kernel_against_ref, get_timing_stats  # type: ignore
from src.utils import set_gpu_arch  # type: ignore


def read_file(path: str) -> str:
    """Load text file and raise on failure."""
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a Triton kernel against a reference implementation.")
    p.add_argument("--ref", required=True, help="Path to reference PyTorch source file")
    p.add_argument("--triton", required=True, help="Path to Triton kernel source file")
    p.add_argument("--device", type=int, default=0, help="CUDA device index to run on (default: 0)")
    p.add_argument("--trials", type=int, default=100, help="Number of performance trials (default: 100)")
    p.add_argument("--verbose", action="store_true", help="Print verbose logs from eval")
    p.add_argument("--json_out", type=str, default=None, help="Optional path to dump result JSON")
    p.add_argument("--gpu_arch", nargs="*", default=["Hopper"], help="TORCH_CUDA_ARCH_LIST override (e.g. Ampere Turing)")
    args = p.parse_args()

    # Basic device / env setup
    torch.cuda.set_device(args.device)
    set_gpu_arch(args.gpu_arch)
    device = torch.device(f"cuda:{args.device}")

    # Read sources
    ref_src    = read_file(args.ref)
    triton_src = read_file(args.triton)

    # Evaluate
    result = eval_kernel_against_ref(
        original_model_src=ref_src,
        custom_model_src=triton_src,
        num_perf_trials=args.trials,
        measure_performance=True,
        verbose=args.verbose,
        backend="triton",
        device=device,
    )

    # Pretty print outcome
    print("\n========= Evaluation Summary =========")
    print(f"Compiled:    {result.compiled}")
    print(f"Correctness: {result.correctness}")
    if result.runtime > 0:
        print(f"Runtime (us): {result.runtime:.2f}")
    if result.runtime_stats:
        stats = result.runtime_stats
        print("Latency stats (us):", {k: round(v, 2) for k, v in stats.items()})

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result.dict(), f, indent=2, default=str)
        print(f"\nDetailed result written to {os.path.abspath(args.json_out)}")


if __name__ == "__main__":
    main() 