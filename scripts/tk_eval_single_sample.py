
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
        self.problem_id = 9
        
        self.verbose = True
        self.log = True

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
        ref_arch_src, new_model_code, verbose=config.verbose, measure_performance=True, num_correct_trials=5, num_perf_trials=100, kernel_dir=kernel_dir
    )
            

            
    print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")

    if config.log:
        with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "a") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(str(kernel_exec_result))



if __name__ == "__main__":
    main()

