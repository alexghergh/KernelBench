import argparse
import modal

# -----------------------------------------------------------------------------
# Script: measure_launch_overhead_modal.py
# -----------------------------------------------------------------------------
# Utility executed with `modal run` to empirically measure the average CUDA
# kernel-launch overhead on a specific GPU type.  Prints the result in µs to
# stdout so it can be copy-pasted into other scripts.
# -----------------------------------------------------------------------------

app = modal.App("kernelbench_launch_probe")
CUDA_TAG = "12.4.0-devel-ubuntu22.04"  

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .apt_install(
        "git",
        "gcc-10",
        "g++-10",
        "clang",
    )
    .pip_install(
        "anthropic",
        "numpy",
        "openai",
        "packaging",
        "pydra_config",
        "torch==2.5.0",
        "tqdm",
        "datasets",
        "transformers",
        "google-generativeai",
        "together",
        "pytest",
        "ninja",
        "utils",
    )
) 

# GPU flavour is overridden via .options(gpu="H100", …) when the remote
# function is invoked from the local entry-point.

def _measure_overhead(trials: int) -> float:
    """Launch an empty CUDA kernel `trials` times and return overhead (µs)."""
    import torch  # imported inside the container

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(trials):
        torch.cuda._sleep(1)
    end.record(); torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)  # milliseconds
    return elapsed_ms * 1000.0 / trials  # microseconds

# ---------------------------------------------------------------------------
# Remote function – runs on the GPU node
# ---------------------------------------------------------------------------

@app.function(image=image, gpu="any", timeout=300)
def launch_probe(trials: int = 10_000) -> float:
    """Return the average CUDA kernel-launch overhead in micro-seconds."""

    return _measure_overhead(trials)

# ---------------------------------------------------------------------------
# Local entry-point – invoked with `modal run` on the developer machine
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(gpu_type: str = "H100", trials: int = 10_000):  # noqa: D401
    """Run the launch-overhead probe on a chosen GPU type via Modal."""

    fn = launch_probe
    if hasattr(fn, "with_options"):
        fn = fn.with_options(gpu=gpu_type)
    elif hasattr(fn, "options"):
        fn = fn.options(gpu=gpu_type)  # older Modal versions

    overhead_us = fn.remote(trials)
    print(f"Average CUDA launch overhead on {gpu_type}: {overhead_us:.2f} µs (over {trials} launches)")

# Allow `python measure_launch_overhead_modal.py` for quick local testing.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_type", "--gpu-type", default="H100")
    parser.add_argument("--trials", type=int, default=10_000)
    cli_args = parser.parse_args()
    main(cli_args.gpu_type, cli_args.trials)

# -------------------------------------------------------------
# Build container image (same deps as other KernelBench Modal scripts)
# -------------------------------------------------------------

# must not exceed host CUDA version

