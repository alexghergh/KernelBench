import os
import pydra
from pydra import REQUIRED, Config

import torch
import numpy as np
from src.eval import (
    load_original_model_and_inputs,
    time_execution_with_cuda_event,
    get_timing_stats,
    set_seed,
    fetch_ref_arch_from_problem_id,
)
from src.dataset import construct_problem_dataset_from_problem_dir
from src.utils import read_file
import os
import json
from tqdm import tqdm


from torch._dynamo import explain
from torch.fx import symbolic_trace


"""
Understand PyTorch Fusion

Usage
# TORCH_LOGS_OUT="test.log" TORCH_LOGS="output_code" python3 ...
# might no need explicit env variable setting

TODO:
- TVM backend <-- this has to be built from source somehow which is not very ideal
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")

TORCH_COMPILE_LOG_DIR = os.path.join(REPO_TOP_PATH, "results", "level_2_torch_compile_logs")

def fetch_ref_arch_from_dataset(dataset: list[str], 
                                problem_id: int) -> tuple[str, str, str]:
    """
    Fetch the reference architecture from the problem directory
    problem_id should be logical index (1-indexed), matching the problem_id in the problem_name

    Returns:
        ref_arch_path: str, the path to the reference architecture
        ref_arch_name: str, the name of the reference architecture
        ref_arch_src: str, the source code of the reference architecture
    """
    ref_arch_path = None
    
    for file in dataset:
        if file.split("/")[-1].split("_")[0] == str(problem_id):
            ref_arch_path = file
            break
    if ref_arch_path is None:
        raise ValueError(f"No reference architecture found for problem_id {problem_id}")
    
    ref_arch_src = read_file(ref_arch_path)

    ref_arch_name = ref_arch_path.split("/")[-1]
    return (ref_arch_path, ref_arch_name, ref_arch_src)


def measure_program_time(
        ref_arch_name: str,
        ref_arch_src: str, 
        num_trials: int = 100,
        use_torch_compile: bool = False,
        torch_compile_backend: str="inductor", 
        torch_compile_options: str="default",
        device: torch.device="cuda:0",
        verbose: bool = False,
) -> dict:
    """
    Measure the time of a KernelBench reference architecture
    """
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    try:
        with torch.no_grad():
            torch.cuda.synchronize(device=device)
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            # Initialize PyTorch model, use this for eager mode execution
            model = Model(*init_inputs)
            
            if use_torch_compile:
                print(f"Using torch.compile to compile model {ref_arch_name} with {torch_compile_backend} backend and {torch_compile_options} mode")
                model = torch.compile(model, backend=torch_compile_backend, mode=torch_compile_options)
            else:
                print(f"Using PyTorch Eager Execution on {ref_arch_name}")
            
            model = model.cuda(device=device)
            torch.cuda.synchronize(device=device)
            elapsed_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=num_trials, verbose=verbose, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)

            if verbose:
                print(f"{ref_arch_name} {runtime_stats}")
            
            return runtime_stats
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")




def get_torch_compiled_model(
        ref_arch_name: str,
        ref_arch_src: str, 
        use_torch_compile: bool = False,
        torch_compile_backend: str="inductor", 
        torch_compile_options: str="default",
        device: torch.device="cuda:0",
        verbose: bool = False,
        log_file: str="torch_compile_output_default.log",
) -> dict:
    """
    Get Torch Compiled Model
    """
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )

    # set environment variables for torch compile logging
    os.environ["TORCH_LOGS"] = "output_code"
    os.environ["TORCH_LOGS_OUT"] = log_file
    
    # Update torch logging settings
    torch._logging.set_logs(output_code=True)
    
    # Optionally, for more detailed logging:
    # set_logs(output_code=True, graph_code=True, aot=True)
    
    # Force reload the logging configuration
    torch._logging._init_logs()


    try:
        with torch.no_grad():
            torch.cuda.synchronize(device=device)
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            # Initialize PyTorch model, use this for eager mode execution
            model = Model(*init_inputs)


            if use_torch_compile:
                print(f"Using torch.compile to compile model {ref_arch_name} with {torch_compile_backend} backend and {torch_compile_options} mode")
                model = torch.compile(model, backend=torch_compile_backend, mode=torch_compile_options)
            else:
                print(f"Using PyTorch Eager Execution on {ref_arch_name}")



            # just run it forward            
            model = model.cuda(device=device)
            torch.cuda.synchronize(device=device)

            # explain the model, haven't figure out how to use this directly
            # explaination = explain(model, *inputs)

            # import pdb; pdb.set_trace()

            elapsed_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=1, verbose=verbose, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)

            if verbose:
                print(f"Torch Compile Log File: {log_file}")

            return model
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")




def test_measure_particular_program(level_num: int, problem_id: int):
    """
    Test measure_program_time on a particular program
    """
    device = torch.device("cuda:0")

    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level_num))
    dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)

    ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)

    log_file = os.path.join(TORCH_COMPILE_LOG_DIR, "{}.log".format(ref_arch_name))
    
    model = get_torch_compiled_model(
        ref_arch_name=ref_arch_name,
        ref_arch_src=ref_arch_src,
        use_torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_options="default",
        device=device,
        verbose=False,
        log_file=log_file)


if __name__ == "__main__":
    # DEBUG and simple testing
    # test_measure_particular_program(2, 28)

    os.makedirs(TORCH_COMPILE_LOG_DIR, exist_ok=True)
    
    test_measure_particular_program(level_num=2, problem_id=80)




