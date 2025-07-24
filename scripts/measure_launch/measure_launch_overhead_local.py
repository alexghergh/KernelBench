#!/usr/bin/env python
"""
measure_launch_overhead_local.py
--------------------------------
Measure average CUDA kernel-launch overhead **on the local machine**.

Usage (PowerShell or Bash):
    python KernelBench/scripts/measure_launch_overhead_local.py --trials 100000

Outputs the GPU name, number of trials, and average micro-seconds per launch.
The result can be fed into `filter_tasks_by_launch_overhead.py` via
  --launch_us <value>
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import torch  # type: ignore
except ImportError as exc:  # pragma: no cover
    sys.exit(
        "[ERROR] PyTorch with CUDA support is required.\n"
        "Install, e.g.:\n"
        "  pip install torch --extra-index-url https://download.pytorch.org/whl/cu121\n"
    )

def measure_launch_overhead(trials: int = 10_000) -> float:
    """Return average GPU-side launch latency in micro-seconds (µs)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot measure launch overhead.")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(trials):
        torch.cuda._sleep(1)  # one clock-cycle no-op kernel
    end.record()
    torch.cuda.synchronize()

    elapsed_ms: float = start.elapsed_time(end)  # milliseconds
    return elapsed_ms * 1_000.0 / trials  # → micro-seconds

def main() -> None:
    parser = argparse.ArgumentParser(description="Measure CUDA kernel-launch overhead (local GPU)")
    parser.add_argument("--trials", type=int, default=10_000, help="Number of empty kernel launches to average")
    args = parser.parse_args()

    t0 = time.perf_counter()
    avg_us = measure_launch_overhead(args.trials)
    dt = time.perf_counter() - t0

    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "N/A"
    print(f"GPU            : {gpu_name}")
    print(f"Trials         : {args.trials:,}")
    print(f"Average launch : {avg_us:.2f} µs")
    print(f"(measurement took {dt:.1f} s)")

if __name__ == "__main__":
    main() 