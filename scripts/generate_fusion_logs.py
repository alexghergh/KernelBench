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
import re


from torch._dynamo import explain
from torch.fx import symbolic_trace


"""
Understand PyTorch Fusion


Usage: Just run python3 generate_fusion_logs.py


# [Deprecate] TORCH_LOGS_OUT="test.log" TORCH_LOGS="output_code" python3 ...
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


class FusionConfig(Config):
    level_num: int = 2
    problem_id: int = 1

    verbose: bool = False

    


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



def extract_fusion_decisions_from_log(log_file: str) -> list[str]:
    """
    Extract fusion decisions from a log file
    Only starts looking for 'Sort' after finding 'def call(args)'
    """
    sort_lines = []
    found_call_def = False
    
    with open(log_file, "r") as f:
        for line in f:
            if "def call(args):" in line:
                found_call_def = True
                continue
                
            if found_call_def and "Sort" in line:
                sort_lines.append(line.strip())
    
    return sort_lines

def get_fused_ops_from_sort_lines(sort_lines: list[str]) -> list[str]:
    """
    Format aten operations from sort lines into clean format
    Input: Lines containing "Original ATen: [aten.op]"
    Output: ['op', '(op1, op2, ...)']
    """
    formatted_ops = []
    

    for line in sort_lines:
        # Use regex to find content after "Original ATen: [...]"
        match = re.search(r"Original ATen: \[(.*?)\]", line)
        if match:
            ops = match.group(1)  # Get the content inside brackets
            # Clean up the operations
            ops = ops.replace('aten.', '')
            # If comma separated, it's a group of operations
            if ',' in ops:
                ops = f"({ops})"
            formatted_ops.append(ops)
    
    return formatted_ops

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
    
    # Delete log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)
        if verbose:
            print(f"Removed existing log file: {log_file}")

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



            elapsed_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=1, verbose=verbose, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)

            if verbose:
                print(f"Torch Compile Log File: {log_file}")

                # to visualize individual fusion decision
                fusion_decisions = extract_fusion_decisions_from_log(log_file)
                print(f"===============Fusion Decision=================================")
                for (i, decision) in enumerate(fusion_decisions):
                    print(f"Fusion Decision {i}: {decision}")

                fused_ops = get_fused_ops_from_sort_lines(fusion_decisions)
                print(f"===============Fused Ops=================================")
                print(f"Fused Ops: {fused_ops}")

            return model
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")



def test_measure_particular_program(level_num: int, problem_id: int, verbose: bool = False):
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
        verbose=verbose,
        log_file=log_file)
    
def generate_fusion_logs_for_all_level_programs(level_num: int):
    """
    Generate fusion logs for all programs in a given level
    """
    device = torch.device("cuda:0")

    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level_num))
    dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)

    num_problems_in_level = len(dataset)
    for problem_id in tqdm(range(1, num_problems_in_level + 1), desc="Generating fusion logs for all Level 2 programs"):
        
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
        
    print(f"Generated fusion logs for all {num_problems_in_level} Level 2 programs")

@pydra.main(base=FusionConfig)
def main(config: FusionConfig):
    level_num = config.level_num
    problem_id = config.problem_id
    verbose = config.verbose

    os.makedirs(TORCH_COMPILE_LOG_DIR, exist_ok=True)    

    # DEBUG: generate a single log file
    # test_measure_particular_program(level_num=level_num, problem_id=problem_id, verbose=verbose)

    # Generate for all Level 2
    generate_fusion_logs_for_all_level_programs(level_num=level_num)

if __name__ == "__main__":
    main()




