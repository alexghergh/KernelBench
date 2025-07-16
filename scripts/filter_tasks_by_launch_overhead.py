import os
import json
import argparse
from typing import Dict, List, Any
import torch

# Resolve repo root and default timing directory early so they are available
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
RESULTS_ROOT_DEFAULT = os.path.join(REPO_ROOT_PATH, "results", "timing")

# ----------------------------------------------------------------------------
# Script: filter_tasks_by_launch_overhead.py
# ----------------------------------------------------------------------------
# Author: KernelBench contributors
#
# Utility to create a reduced KernelBench task list whose reference runtimes are
# large enough that CUDA kernel-launch overhead is negligible.
#
# For each GPU timing directory (e.g. A100_modal, H100_together, …) the script
# reads the chosen baseline timing file, compares each problem's mean runtime
# against a user-defined threshold, and writes a JSON file containing the
# problems that pass the threshold.
#
# Threshold options
# -----------------
#   1. Specify launch overhead (in microseconds) *and* a multiplier factor.
#      threshold_ms = launch_us * factor / 1000.
#   2. Or give --threshold_ms directly to override the computed value.
#
# Example usages
# --------------
#   # Keep problems whose baseline ≥ 100×8 µs = 0.8 ms (≈1 % overhead)
#   python filter_tasks_by_launch_overhead.py \
#       --gpu_dirs A100_modal A100-80GB_modal B200_together H100_together \
#       --launch_us 8 --factor 100
#
#   # Same but allow quick 5 % tolerance (20×)
#   python filter_tasks_by_launch_overhead.py --factor 20
#
#   # Provide an explicit threshold in ms (overrides launch_us & factor)
#   python filter_tasks_by_launch_overhead.py --threshold_ms 1.0
# ----------------------------------------------------------------------------

def load_baseline_json(path: str) -> Dict[str, Any]:
    """Read a timing JSON file and return its dict."""
    with open(path, "r") as f:
        return json.load(f)


def filter_problems(baseline: Dict[str, Any], threshold_ms: float, *, keep_times: bool) -> Dict[str, Any]:
    """Return per-level mapping of problems that pass threshold.

    If `keep_times` is True the value is the *full stats* dict for each
    filename; otherwise the value is just a filename list for backward
    compatibility.
    """
    out: Dict[str, Any] = {}
    for level, problems in baseline.items():
        selected = {
            name: stats for name, stats in problems.items() if stats.get("mean", 0) >= threshold_ms
        }
        if not selected:
            continue

        if keep_times:
            out[level] = dict(sorted(selected.items()))
        else:
            out[level] = sorted(selected)
    return out


def measure_launch_overhead(num_trials: int = 10_000) -> float:
    """Empirically measure average CUDA kernel-launch overhead (µs).

    The measurement launches an empty `_sleep(1)` kernel many times and
    divides the total elapsed time by the number of launches.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot measure launch overhead.")

    # Ensure we use the current device (default: cuda:0)
    device = torch.cuda.current_device()

    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    torch.cuda.synchronize(device)
    start.record()
    for _ in range(num_trials):
        # `_sleep(1)` is essentially a no-op kernel that occupies the GPU for
        # one clock cycle – perfect for measuring dispatch overhead.
        torch.cuda._sleep(1)
    end.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)  # milliseconds
    avg_us = elapsed_ms * 1000.0 / num_trials
    return avg_us


# -----------------------------------------------------------------------------
# Optional: Modal support to measure launch overhead on cloud GPUs
# -----------------------------------------------------------------------------
try:
    import modal  # type: ignore
    HAVE_MODAL = True
except ImportError:
    HAVE_MODAL = False


if HAVE_MODAL:
    modal_app = modal.App("measure_launch_overhead")

    @modal_app.cls(gpu="any", timeout=300)
    class ModalProbe:
        @modal.method()
        def measure(self, trials: int, gpu_type: str) -> float:
            import torch
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(trials):
                torch.cuda._sleep(1)
            end.record(); torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            return elapsed_ms * 1000.0 / trials


def measure_launch_overhead_modal(trials: int, gpu_type: str) -> float:
    """Measure launch overhead on a specific GPU via Modal."""
    if not HAVE_MODAL:
        raise RuntimeError("Modal is not installed. Install modal or run without --use_modal.")

    import modal

    # Create a small Modal app dynamically so we can parameterize the GPU type.
    app_name = f"launch_probe_{gpu_type}" if gpu_type != "any" else "launch_probe_any"
    stub = modal.App(app_name)

    @stub.function(gpu=gpu_type if gpu_type != "any" else "any", timeout=300)
    def _probe(trials: int):
        import torch
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(trials):
            torch.cuda._sleep(1)
        end.record(); torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        return elapsed_ms * 1000.0 / trials

    # Run the probe inside a Modal execution context.
    with stub.run():
        return _probe.remote(trials)


def main():
    parser = argparse.ArgumentParser(description="Filter KernelBench tasks based on launch-overhead tolerance")
    parser.add_argument("--gpu_dirs", nargs="*", default=[
        "A100_modal",
        "A100-80GB_modal",
        "B200_together",
        "H100_together",
    ], help="List of sub-directories inside results/timing to process")
    parser.add_argument("--baseline_fname", default="baseline_time_torch.json", help="Primary baseline timing filename (e.g. eager)")
    parser.add_argument("--secondary_baseline_fname", default="baseline_time_torch_compile_inductor_reduce-overhead.json", help="Secondary baseline timing file (e.g. reduce-overhead). Used only when --require_both is set.")
    parser.add_argument("--require_both", action="store_true", help="If set, a task must satisfy the threshold in BOTH baseline files to be kept.")
    parser.add_argument("--results_root", default=RESULTS_ROOT_DEFAULT, help="Root directory where timing sub-dirs live")
    parser.add_argument("--launch_us", type=float, default=8.0, help="Measured kernel-launch overhead in microseconds")
    parser.add_argument("--factor", type=float, default=100.0, help="Multiplier; runtime must be ≥ factor×launch_us")
    parser.add_argument("--threshold_ms", type=float, default=None, help="Explicit threshold in milliseconds (overrides launch_us*factor)")
    parser.add_argument("--output_dir", default="filtered_tasklists", help="Where to write the resulting JSON files")
    parser.add_argument("--include_times", action="store_true", help="If set, store mean runtime alongside each problem name in output JSONs")
    parser.add_argument("--measure_launch_overhead", action="store_true", help="Empirically measure kernel-launch overhead and override --launch_us")
    parser.add_argument("--launch_trials", type=int, default=10_000, help="Number of empty-kernel launches used when measuring overhead")
    parser.add_argument("--use_modal", action="store_true", help="Measure launch overhead on a Modal GPU (requires modal package and credentials)")
    parser.add_argument("--modal_gpu", type=str, default="any", help="GPU type to request on Modal (e.g., H100, B200)")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Optionally measure launch overhead in the current runtime environment
    # ---------------------------------------------------------------------
    if args.measure_launch_overhead:
        try:
            if args.use_modal:
                measured = measure_launch_overhead_modal(args.launch_trials, args.modal_gpu)
                src = f"Modal remote GPU (GPU type: {args.modal_gpu})"
            else:
                measured = measure_launch_overhead(args.launch_trials)
                src = "local GPU"
            print(f"[INFO] Measured launch overhead on {src}: {measured:.2f} µs over {args.launch_trials} trials")
            args.launch_us = measured
        except Exception as e:
            print(f"[WARN] Failed to measure launch overhead automatically: {e}. Using provided --launch_us={args.launch_us} µs.")

    # Determine threshold in ms
    threshold_ms = args.threshold_ms if args.threshold_ms is not None else (args.launch_us * args.factor / 1000.0)
    os.makedirs(args.output_dir, exist_ok=True)

    summary: Dict[str, Dict[str, List[str]]] = {}

    for gpu_dir in args.gpu_dirs:
        primary_path = os.path.join(args.results_root, gpu_dir, args.baseline_fname)
        if not os.path.exists(primary_path):
            print(f"[WARN] Missing {primary_path}")
            continue
        baseline_primary = load_baseline_json(primary_path)

        if args.require_both:
            secondary_path = os.path.join(args.results_root, gpu_dir, args.secondary_baseline_fname)
            if not os.path.exists(secondary_path):
                print(f"[WARN] Missing secondary baseline {secondary_path}; skipping GPU dir {gpu_dir}")
                continue
            baseline_secondary = load_baseline_json(secondary_path)

            # keep problem only if it appears in both and satisfies threshold in both
            filtered_primary = filter_problems(baseline_primary, threshold_ms, keep_times=args.include_times)
            filtered_secondary = filter_problems(baseline_secondary, threshold_ms, keep_times=args.include_times)

            filtered = {}
            for level in filtered_primary:
                if level in filtered_secondary:
                    kept = sorted(list(set(filtered_primary[level]).intersection(filtered_secondary[level])))
                    if kept:
                        filtered[level] = kept
        else:
            filtered = filter_problems(baseline_primary, threshold_ms, keep_times=args.include_times)

        summary[gpu_dir] = filtered

        # compute excluded tasks
        excluded: Dict[str, Any] = {}
        for level, problems in baseline_primary.items():
            kept_names = (
                set(filtered.get(level, {}).keys()) if args.include_times else set(filtered.get(level, []))
            )

            if args.include_times:
                excluded[level] = {
                    name: stats for name, stats in problems.items() if name not in kept_names
                }
                if not excluded[level]:
                    excluded.pop(level)
            else:
                excl_list = [name for name in problems if name not in kept_names]
                if excl_list:
                    excluded[level] = sorted(excl_list)

        # Write per-GPU file
        out_path = os.path.join(args.output_dir, f"{gpu_dir}_filtered_{threshold_ms:.3f}ms.json")
        with open(out_path, "w") as f:
            json.dump(filtered, f, indent=2)
        print(f"[INFO] {gpu_dir}: kept {sum(len(v) for v in filtered.values())} problems → {out_path}")

        excluded_path = os.path.join(args.output_dir, f"{gpu_dir}_excluded_{threshold_ms:.3f}ms.json")
        with open(excluded_path, "w") as f:
            json.dump(excluded, f, indent=2)
        print(f"[INFO] {gpu_dir}: excluded {sum(len(v) for v in excluded.values())} problems → {excluded_path}")

    # Write aggregate
    agg_path = os.path.join(args.output_dir, f"aggregate_filtered_{threshold_ms:.3f}ms.json")
    with open(agg_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Aggregate written to {agg_path}")


if __name__ == "__main__":
    main() 