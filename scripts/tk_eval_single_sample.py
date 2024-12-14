
import pydra
from pydra import REQUIRED, Config
import os, sys
from src.dataset import construct_problem_dataset_from_problem_dir
import torch
import json

import subprocess

import shutil
from src.eval import KernelExecResult, eval_kernel_against_ref, check_metadata_serializable_all_types
# ThunderKitten specific prompts, edit there!
from src.utils import read_file



REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)



class EvalConfig(Config):

    def __init__(self, **kwargs):
        self.dataset_src = "local_tk" # either huggingface or local

        self.eval_mode = "local"


        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.kernel_builds_dir = os.path.join(REPO_TOP_DIR, "kernels")

        self.level = 1
        self.problem_id = 8
        
        self.verbose = True
        self.log = True

        self.fetch_baseline_time = True
        self.baseline_time_dir = os.path.join(REPO_TOP_DIR, "results/timing/tk")


def fetch_baseline_time(
    level_name: str, problem_id: int, problem_name: str, baseline_time_filepath: str
) -> dict:
    """
    Fetch the baseline time from the time
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    baseline_time = baseline_json[level_name].get(problem_name, None)
    return baseline_time

def create_compilation_file_kernel_exec_result(error_msg):
    """
    Create a KernelExecResult to signal failure for when we fail to compile the kernel
    """
    return KernelExecResult(
        compiled=False,
        correctness=False,
        runtime=-1.0,
        runtime_stats={},
        metadata={"error": check_metadata_serializable_all_types({"error": error_msg})}
    )


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):

    print(f"Starting Eval with config: {config}")

    kernel_dir = os.path.join(config.kernel_builds_dir, f"level_{config.level}_problem_{config.problem_id}")

    # asume kernel is built already and ideally compilable
    # Load in compiled custom kernel module
    sys.path.append(kernel_dir) # point to the specific binary in the directory
    try:
        import tk_kernels # we name all the thunderkitten kernel modules as tk_kernels now!
        print(f"Imported ThunderKittens Kernel modules at {kernel_dir}: {dir(tk_kernels)}")
    except ImportError as e:
        print(f"Failed to import tk_kernels: {e}")
        kernel_exec_result = create_compilation_file_kernel_exec_result(f"Failed to import tk_kernels: {e}")



    # fetch reference architecture
    if config.dataset_src == "local_tk":
        curr_level_dataset = construct_problem_dataset_from_problem_dir(os.path.join(REPO_TOP_DIR, "KernelBench", f"tk"))
        for problem_path in curr_level_dataset:
            problem_name = os.path.basename(problem_path)
            problem_number = int(problem_name.split("_")[0])
            if problem_number == config.problem_id:
                ref_arch_path = problem_path
                break
        assert ref_arch_path is not None, f"Problem {config.problem_id} not found in dataset"
        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)
    else:
        raise NotImplementedError(f"Dataset source {config.dataset_src} not supported")

    new_model_code = read_file(os.path.join(config.logdir, f"new_model_code_level_{config.level}_problem_{config.problem_id}.py"))  

    ############################################################
    # Start Evaluation
    # Evaluate the kernel, against original reference
    # NOTE: no need to wrap around process here as only a single sample
    # see batch eval for examples of process isolation
    ############################################################
   

    kernel_exec_result = None

    # new model code
    kernel_exec_result = eval_kernel_against_ref(
        ref_arch_src, new_model_code, verbose=config.verbose, measure_performance=True, num_correct_trials=1, num_perf_trials=100, kernel_dir=kernel_dir
    )

    tk_time = kernel_exec_result.runtime_stats 
            
    print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")

    if config.log:
        with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "a") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(str(kernel_exec_result))


    if config.fetch_baseline_time:

        baseline_time = fetch_baseline_time(f"level_tk", config.problem_id, problem_name, os.path.join(config.baseline_time_dir, f"baseline_time.json"))
        baseline_torch_compile_time = fetch_baseline_time(f"level_tk", config.problem_id, problem_name, os.path.join(config.baseline_time_dir, f"baseline_time_torch_compile.json"))
        baselien_torch_compile_max_autotune_time = fetch_baseline_time(f"level_tk", config.problem_id, problem_name, os.path.join(config.baseline_time_dir, f"baseline_time_torch_compile_mode_max-autotune.json"))
        print(f"""Baseline time for problem {problem_name}:
                Torch Eager: {baseline_time}
                Torch Compile: {baseline_torch_compile_time}
                Torch Compile Max Autotune: {baselien_torch_compile_max_autotune_time}""")

    print(f"TK Time: {tk_time}")

    # Create a table comparing all timing results
    print("\nTiming Comparison Table (all times in ms):")
    print("-" * 80)
    print(f"{'Method':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Trials':>8}")
    print("-" * 88)
    
    if tk_time:
        print(f"{'TK Implementation':<30} {tk_time['mean']:>10.3f} {tk_time['std']:>10.3f} {tk_time['min']:>10.3f} {tk_time['max']:>10.3f} {tk_time['num_trials']:>8}")
    
    if baseline_time:
        print(f"{'Torch Eager':<30} {baseline_time['mean']:>10.3f} {baseline_time['std']:>10.3f} {baseline_time['min']:>10.3f} {baseline_time['max']:>10.3f} {baseline_time['num_trials']:>8}")
    
    if baseline_torch_compile_time:
        print(f"{'Torch Compile':<30} {baseline_torch_compile_time['mean']:>10.3f} {baseline_torch_compile_time['std']:>10.3f} {baseline_torch_compile_time['min']:>10.3f} {baseline_torch_compile_time['max']:>10.3f} {baseline_torch_compile_time['num_trials']:>8}")
    
    if baselien_torch_compile_max_autotune_time:
        print(f"{'Torch Compile Max Autotune':<30} {baselien_torch_compile_max_autotune_time['mean']:>10.3f} {baselien_torch_compile_max_autotune_time['std']:>10.3f} {baselien_torch_compile_max_autotune_time['min']:>10.3f} {baselien_torch_compile_max_autotune_time['max']:>10.3f} {baselien_torch_compile_max_autotune_time['num_trials']:>8}")
    print("-" * 80)

if __name__ == "__main__":
    main()

