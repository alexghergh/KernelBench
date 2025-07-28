"""
run_triton_generation.py
========================
Generate Triton kernels with an LLM (o3) for every problem in KernelBench, evaluate them, and log speed-ups over the reference PyTorch implementation.

Usage (example):
    python scripts/run_triton_generation.py --level 1 --run_name my_first_run \
        --server_type deepseek --model_name deepseek-coder --temperature 0.2 \
        --num_workers 4

Outputs in results/triton_runs/<run_name>/
    ├─ results.jsonl     # newline-delimited JSON with metadata per successful kernel
    └─ kernels/          # *.py files for each generated kernel
"""
import os
import json
from dataclasses import dataclass
from typing import List
import traceback

import torch
import modal
from pydra import Config, REQUIRED, main as pydra_main
from dotenv import load_dotenv, find_dotenv
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Load environment variables (e.g. OPENAI_API_KEY) from the nearest .env file
# This lets the script run on machines where the key is stored in the project
# root rather than exported in the shell.
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv(), override=False)

from src.dataset import construct_kernelbench_dataset
from src.prompt_constructor_triton import prompt_generate_custom_triton_from_prompt_template
from src.utils import (
    create_inference_server_from_presets,
    extract_first_code,
    read_file,
    maybe_multithread,
    set_gpu_arch,
)
from src.eval import (
    eval_kernel_against_ref,
    load_original_model_and_inputs,
    time_execution_with_cuda_event,
    get_timing_stats,
)


REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


gpu_arch_mapping = {
    "L40S": ["Ada"],
    "H100": ["Hopper"],
    "A100": ["Ampere"],
    "L4": ["Ada"],
    "T4": ["Turing"],
    "A10G": ["Ampere"],
}


class RunConfig(Config):
    def __init__(self):
        # Dataset options
        self.dataset_src = "local"  # or "huggingface"
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED  # KernelBench level to process (int)
        self.subset = (None, None)  # (start_id, end_id) if you want a slice
        # Inference options
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.temperature = 0.0
        self.max_tokens = 8192
        self.num_workers = 1
        self.api_query_interval = 10  # seconds between requests when multithreading
        # GPU / eval
        self.gpu_device = 0
        self.gpu_arch = ["Hopper"]  # set empty list to build for all archs
        # Execution target
        self.eval_mode = "local"        # "local" or "modal"
        self.gpu = "H100"                # modal GPU type when eval_mode=="modal"

        # Logging & storage
        self.run_name = REQUIRED  # results directory under results/triton_runs/
        self.results_root = os.path.join(REPO_TOP_DIR, "results", "triton_runs")
        self.verbose = False
        # Performance measurement
        self.num_perf_trials = 100
        # Retry settings
        self.max_retries = 3   # number of times to re-generate/evaluate a problem on failure

    def __repr__(self):
        return f"RunConfig({self.to_dict()})"


@dataclass
class WorkItem:
    problem_id: int


def measure_reference_runtime(ref_arch_src: str, device: torch.device, num_trials: int = 100):
    """Return mean runtime (µs) of the reference PyTorch model."""
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(ref_arch_src, context)
    with torch.no_grad():
        init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in get_init_inputs()]
        model = Model(*init_inputs).cuda(device=device)
    inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
    torch.cuda.synchronize(device)
    elapsed_times = time_execution_with_cuda_event(
        model, *inputs, num_trials=num_trials, verbose=False, device=device
    )
    stats = get_timing_stats(elapsed_times, device=device)
    return stats["mean"], stats  # mean µs, full stats dict


# ---------------------------------------------------------------------------
# Modal image & remote evaluator
# ---------------------------------------------------------------------------

cuda_version = "12.4.0"  # should be ≤ host CUDA version
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
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
        "python-dotenv",
        "triton",  # new dependency for Triton kernels
    )
    .env({"FORCE_REBUILD_V2": "1"})
    # Copy KernelBench codebase and its src package into the container so `import src` works
    .add_local_dir(os.path.join(REPO_TOP_DIR, "KernelBench"), remote_path="/root/KernelBench")
    .add_local_dir(os.path.join(REPO_TOP_DIR, "src"), remote_path="/root/src")
)

app = modal.App("run_triton_generationV2")


@app.cls(image=image)
class TritonEval:

    @modal.method()
    def eval_triton_kernel(self, ref_arch_src: str, kernel_code: str, gpu_arch: list[str], num_perf_trials: int, verbose: bool = False):
        """Compile & time the Triton kernel plus reference model on a cloud GPU."""
        import torch
        import inspect
        from src.utils import set_gpu_arch as _set_gpu_arch
        from src.eval import (
            eval_kernel_against_ref,
            load_original_model_and_inputs,
            time_execution_with_cuda_event,
            get_timing_stats,
        )

        _set_gpu_arch(gpu_arch)
        device = torch.device("cuda:0")

        # Evaluate Triton kernel
        triton_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_code,
            verbose=verbose,
            measure_performance=True,
            num_perf_trials=num_perf_trials,
            backend="triton",
            device=device,
        )

        # Measure reference runtime only if Triton compiled & passed correctness
        ref_runtime_stats = {}
        ref_runtime_us = None
        if triton_result.compiled and triton_result.correctness:
            context = {}
            # Execute the reference source code to populate the context
            exec(ref_arch_src, context)
            
            # Extract the functions and Model class that were just defined
            Model = context['Model']
            get_init_inputs = context['get_init_inputs']
            get_inputs = context['get_inputs']

            with torch.no_grad():
                # Dynamically determine the required arguments for get_init_inputs
                sig_init = inspect.signature(get_init_inputs)
                init_args = {param: context[param] for param in sig_init.parameters}
                
                # Call the function with the discovered arguments
                init_inputs_tuple = get_init_inputs(**init_args)
                init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs_tuple]
                
                model = Model(*init_inputs).cuda(device=device)
            
            # Do the same for get_inputs
            sig_inputs = inspect.signature(get_inputs)
            run_args = {param: context[param] for param in sig_inputs.parameters}
            inputs_tuple = get_inputs(**run_args)
            inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs_tuple]
            # --- END OF THE CRUCIAL FIX ---

            torch.cuda.synchronize(device)
            elapsed = time_execution_with_cuda_event(model, *inputs, num_trials=num_perf_trials, verbose=False, device=device)
            ref_runtime_stats = get_timing_stats(elapsed, device=device)
            ref_runtime_us = ref_runtime_stats.get("mean") # Use .get() for safety

        speedup = None
        if ref_runtime_us and triton_result.runtime > 0:
            speedup = ref_runtime_us / triton_result.runtime

        return {
            "triton_result": triton_result.dict(),
            "ref_runtime_stats": ref_runtime_stats,
            "speedup": speedup,
        }


def process_single_problem(work: WorkItem, config: RunConfig, dataset, inference_server, out_kernel_dir, jsonl_path):
    for attempt in range(1, config.max_retries + 1):
        try:
            # Fetch problem source (outside retry loop is fine but cheap)
            if config.dataset_src == "huggingface":
                row = dataset.filter(lambda x: x["problem_id"] == work.problem_id, desc=None)
                ref_arch_src = row["code"][0]
                problem_name = row["name"][0]
            else:
                ref_arch_path = dataset[work.problem_id - 1]
                ref_arch_src = read_file(ref_arch_path)
                problem_name = os.path.basename(ref_arch_path)

            # Build prompt & generate kernel
            prompt = prompt_generate_custom_triton_from_prompt_template(ref_arch_src)
            try:
                kernel_code_raw = inference_server(prompt)
                if config.verbose:
                    print(f"[debug] type(kernel_code_raw)={type(kernel_code_raw)}")
                    print(f"[debug] head:\n{str(kernel_code_raw)[:200]}")
            except Exception:
                print("[GENERATION] failed:")
                traceback.print_exc()
                raise

            try:
                kernel_code = extract_first_code(kernel_code_raw, ["python", "triton", "cpp", "ptx"])
            except Exception:
                print("[EXTRACTION] failed:")
                traceback.print_exc()
                raise

            # --------------------------------------------------
            # Evaluate kernel
            # --------------------------------------------------
            success = False
            if config.eval_mode == "modal":
                remote_res = TritonEval.with_options(gpu=config.gpu)().eval_triton_kernel.remote(
                    ref_arch_src,
                    kernel_code,
                    gpu_arch_mapping.get(config.gpu, config.gpu_arch),
                    config.num_perf_trials,
                    config.verbose,
                )

                # Block and wait for the remote job to finish

                # Now `remote_res` is the actual dictionary returned from the function
                triton_dict = remote_res["triton_result"]
                success = triton_dict["compiled"] and triton_dict["correctness"]

                if success:
                    triton_runtime_us = triton_dict["runtime"]
                    ref_runtime_us = remote_res["ref_runtime_stats"].get("mean")
                    ref_stats = remote_res["ref_runtime_stats"]
                    speedup = remote_res["speedup"]
                    result_runtime_stats = triton_dict["runtime_stats"]
            else:
                result = eval_kernel_against_ref(
                    original_model_src=ref_arch_src,
                    custom_model_src=kernel_code,
                    verbose=config.verbose,
                    measure_performance=True,
                    num_perf_trials=config.num_perf_trials,
                    backend="triton",
                )
                success = result.compiled and result.correctness

                if success:
                    triton_runtime_us = result.runtime
                    ref_runtime_us, ref_stats = measure_reference_runtime(
                        ref_arch_src, torch.device(f"cuda:{config.gpu_device}"), num_trials=config.num_perf_trials
                    )
                    speedup = ref_runtime_us / triton_runtime_us if triton_runtime_us > 0 else None
                    result_runtime_stats = result.runtime_stats

            if success:
                # Save kernel and record results
                kernel_filename = f"level{config.level}_problem{work.problem_id}_triton.py"
                kernel_path = os.path.join(out_kernel_dir, kernel_filename)
                with open(kernel_path, "w") as f:
                    f.write(kernel_code)

                record = dict(
                    level=config.level,
                    problem_id=work.problem_id,
                    problem_name=problem_name,
                    runtime_original_us=ref_runtime_us,
                    runtime_triton_us=triton_runtime_us,
                    speedup=speedup,
                    kernel_path=kernel_path,
                    runtime_original_stats=ref_stats,
                    runtime_triton_stats=result_runtime_stats,
                    attempts=attempt,
                )
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                if config.verbose:
                    print(f"[Success] Problem {work.problem_id} after {attempt} attempt(s)")
                return True

            else:
                if config.verbose:
                    print(f"[Retry] Problem {work.problem_id} attempt {attempt}/{config.max_retries} failed")

        except Exception as e:
            if config.verbose:
                print(f"[Error] Problem {work.problem_id} attempt {attempt}: {e}")
                print(traceback.format_exc())

    # All retries exhausted
    return False


@pydra_main(base=RunConfig)
def main(config: RunConfig):
    print(f"Starting Triton generation run with config: {config}")

    # Dataset selection
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)[f"level_{config.level}"]
    else:
        dataset = construct_kernelbench_dataset(config.level)

    # Determine problem range
    total = len(dataset)
    if config.subset == (None, None):
        ids = range(1, total + 1)
    else:
        ids = range(config.subset[0], config.subset[1] + 1)

    # Prepare GPU context (only if we intend to run locally)
    if config.eval_mode == "local":
        # Will fail on CPU-only hosts, so guard with eval_mode check
        import torch.cuda as _cuda
        if not _cuda.is_available():
            raise RuntimeError("CUDA not available locally but eval_mode=local; either switch to modal or run on a GPU host.")
        torch.cuda.set_device(config.gpu_device)
        set_gpu_arch(config.gpu_arch)

    # Prepare output dirs
    run_dir = os.path.join(config.results_root, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    out_kernel_dir = os.path.join(run_dir, "kernels")
    os.makedirs(out_kernel_dir, exist_ok=True)
    jsonl_path = os.path.join(run_dir, "results.jsonl")

    # Inference server
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
    )

    # Work list
    work_items = [WorkItem(problem_id=i) for i in ids]

    # Run in parallel if requested; ensure Modal app is started only once
    def _run_batch():
        maybe_multithread(
            process_single_problem,
            work_items,
            num_workers=config.num_workers,
            time_interval=config.api_query_interval,
            # extra args
            config=config,
            dataset=dataset,
            inference_server=inference_server,
            out_kernel_dir=out_kernel_dir,
            jsonl_path=jsonl_path,
        )

    if config.eval_mode == "modal":
        import modal
        modal.enable_output()
        with app.run():
            _run_batch()
    else:
        _run_batch()

    print(f"Finished run. Results saved to {jsonl_path}")


if __name__ == "__main__":
    main() 