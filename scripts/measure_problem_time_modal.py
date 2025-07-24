import os
import json
import argparse
import time

import modal
import torch

from src.eval import (
    load_original_model_and_inputs,
    time_execution_with_cuda_event,
    get_timing_stats,
    set_seed,
)
from src.dataset import construct_problem_dataset_from_problem_dir
from src.utils import read_file

################################################################################
# Paths
################################################################################

# The repository top path is one directory up from this script
REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")
SRC_PATH = os.path.join(REPO_TOP_PATH, "src")

################################################################################
# Modal Image & App definition (re-use config from generate_baseline_time_modal.py)
################################################################################

gpu_arch_mapping = {
    "L40S": ["Ada"],
    "H100": ["Hopper"],
    "A100": ["Ampere"],
    "A100-80GB": ["Ampere"],
    "L4": ["Ada"],
    "T4": ["Turing"],
    "A10G": ["Ampere"],
}

# Modal CUDA image parameters – keep in sync with generate_baseline_time_modal.py
CUDA_VERSION = "12.4.0"  # should be <= host CUDA version
FLAVOR = "devel"  # includes full CUDA toolkit
OPERATING_SYS = "ubuntu22.04"
TAG = f"{CUDA_VERSION}-{FLAVOR}-{OPERATING_SYS}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.10")
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
        "python-dotenv",  # needed to load environment variables via src.utils
        "utils",
        "einops",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(KERNEL_BENCH_PATH, remote_path="/root/KernelBench")
    .add_local_dir(SRC_PATH, remote_path="/root/src")
)

################################################################################
# Helper Functions
################################################################################

def fetch_ref_arch_from_dataset(dataset: list[str], problem_id: int):
    """Return (ref_arch_path, ref_arch_name, ref_arch_src) for the given problem id."""
    ref_arch_path = None
    for file_path in dataset:
        filename = os.path.basename(file_path)
        if filename.split("_")[0] == str(problem_id):
            ref_arch_path = file_path
            break
    if ref_arch_path is None:
        raise ValueError(f"No reference architecture found for problem_id {problem_id}")
    ref_arch_src = read_file(ref_arch_path)
    ref_arch_name = os.path.basename(ref_arch_path)
    return ref_arch_path, ref_arch_name, ref_arch_src

################################################################################
# Modal remote class
################################################################################

app = modal.App("measure_problem_modal")


@app.cls(image=image, scaledown_window=5)
class EvalFunc:
    @modal.method()
    def measure_program_time(
        self,
        ref_arch_name: str,
        ref_arch_src: str,
        num_trials: int = 100,
        use_torch_compile: bool = False,
        torch_compile_backend: str = "inductor",
        torch_compile_options: str = "default",
        verbose: bool = True,
    ):
        """Measure execution time for a single KernelBench reference architecture."""
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(ref_arch_src, {})

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                torch.cuda.synchronize(device=device)
                set_seed(42)
                inputs = get_inputs()
                set_seed(42)
                init_inputs = get_init_inputs()

                inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
                init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

                model = Model(*init_inputs)

                if use_torch_compile:
                    if verbose:
                        print(
                            f"[Eval] torch.compile model {ref_arch_name} (backend={torch_compile_backend}, mode={torch_compile_options})"
                        )
                    model = torch.compile(
                        model,
                        backend=torch_compile_backend,
                        mode=torch_compile_options,
                    )
                else:
                    if verbose:
                        print(f"[Eval] PyTorch eager execution for {ref_arch_name}")

                model = model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                elapsed_times = time_execution_with_cuda_event(
                    model, *inputs, num_trials=num_trials, verbose=verbose, device=device
                )
                stats = get_timing_stats(elapsed_times, device=device)
                if verbose:
                    print(f"[Result] {ref_arch_name}: {stats}")
                return stats
        except Exception as e:
            print(f"[ERROR] Measuring performance failed for {ref_arch_name}: {e}")
            return None

################################################################################
# Wrapper
################################################################################

def measure_program_time_wrapper(*args, **kwargs):
    """Utility wrapper to execute EvalFunc.measure_program_time inside a Modal app run."""
    with app.run():
        # Default to one GPU unless overridden via kwargs (Modal's API)
        gpu_type = kwargs.pop("gpu", "H100")
        return EvalFunc.with_options(gpu=gpu_type)().measure_program_time.remote(*args, **kwargs)

################################################################################
# Command Line Interface
################################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Measure timing for a single KernelBench problem using Modal.")

    parser.add_argument("--level", type=int, required=True, help="KernelBench level (e.g., 1, 2, 3, 4)")
    parser.add_argument("--problem-id", type=int, help="Single problem ID within the level (matches prefix in filename)")
    parser.add_argument("--problem-ids", type=str, help="Comma-separated list of problem IDs (e.g. '12,14,15')")
    parser.add_argument("--all", action="store_true", help="Measure every problem in the specified level")
    parser.add_argument("--trials", type=int, default=100, help="Number of timing trials to run")
    parser.add_argument("--torch-compile", action="store_true", help="Use torch.compile instead of eager")
    parser.add_argument("--backend", default="inductor", help="Backend for torch.compile (default: inductor)")
    parser.add_argument("--mode", default="default", help="Mode for torch.compile (default: default)")
    parser.add_argument("--gpu", default="H100", help="GPU type for Modal (default: H100)")

    args = parser.parse_args()

    # Build list of problem IDs
    if args.all:
        # Collect all problem ids present in the level directory
        problem_ids = []  # will be filled after dataset creation
    elif args.problem_ids:
        problem_ids = [int(tok) for tok in args.problem_ids.split(',') if tok.strip()]
    else:
        if args.problem_id is None:
            parser.error("Must specify --problem-id, --problem-ids, or --all")
        problem_ids = [args.problem_id]

    # Build dataset once
    level_dir = os.path.join(KERNEL_BENCH_PATH, f"level{args.level}")
    if not os.path.isdir(level_dir):
        raise FileNotFoundError(f"Level directory not found: {level_dir}")

    dataset = construct_problem_dataset_from_problem_dir(level_dir)

    # If --all flag was used, derive problem_ids from dataset filenames
    if args.all:
        problem_ids = [int(os.path.basename(p).split("_")[0]) for p in dataset]

    aggregated_results = {}

    for pid in problem_ids:
        try:
            _, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, pid)
        except ValueError as e:
            print(f"[WARN] {e}")
            aggregated_results[str(pid)] = None
            continue

        print(f"\n--- Measuring problem {pid} ({ref_arch_name}) ---")

        result = measure_program_time_wrapper(
            ref_arch_name,
            ref_arch_src,
            args.trials,
            args.torch_compile,
            args.backend,
            args.mode,
            True,  # verbose
            gpu=args.gpu,
        )

        aggregated_results[str(pid)] = result

    # Pretty print aggregated results
    print("\n======================= Timing Results =======================")
    print(json.dumps(aggregated_results, indent=4))
    print("===========================================================\n")


if __name__ == "__main__":
    main() 