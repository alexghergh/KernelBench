import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json

import subprocess
from datasets import load_dataset

import shutil
from src.dataset import construct_kernelbench_dataset
from src.eval import KernelExecResult, eval_kernel_against_ref, check_metadata_serializable_all_types
# ThunderKitten specific prompts, edit there!
from src.prompt_constructor import prompt_generate_custom_thunderkitten_from_prompt_template
from src.utils import extract_code_blocks, extract_code_blocks_of_type, extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets, create_tk_makefile


"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging


Speically modified for ThunderKitten
The pipeline is: Construct Prompt -> Inference -> Extract .cu (TK) code + makefile -> Compile + Build -> Evaluate

TODO:
- Experient with TK Prompt
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

class EvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"


        # Problem Specification
        self.level = REQUIRED
        # NOTE: this is the logical index (problem id the problem_name)\
        self.problem_id = REQUIRED

        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "local"

        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.0

        # Build + Evaluation config
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ["Hopper"]


        # For ThunderKitten, we need to build the Kernel locally binary (not inline cuda)
        # let's specify where to write and build the kernels
        self.kernel_builds_dir = os.path.join(REPO_TOP_DIR, "kernels")
        self.clean_kernel_build = False # remove any previous built binary
        
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False

        self.log = False
        self.log_prompt = False
        self.log_generated_kernel = False
        self.log_eval_result = False

        self.stop_before_eval = False # for debugging
        self.skip_generation = False # for debugging
        

    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"

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
    """
    Keep it simple: Generate and evaluate a single sample
    """
    print(f"Starting Eval with config: {config}")

    # Configurations
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    assert os.environ.get("THUNDERKITTENS_ROOT"), "THUNDERKITTENS_ROOT environment variable is not set, please run source env.src in the ThunderKitten repo"
    
    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)  # otherwise build for all architectures

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)
        
    # Problem Checks
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"


    # # 1. Fetch Problem
    # if config.dataset_src == "huggingface":
    #     curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
    #     ref_arch_src = curr_problem_row["code"][0]
    #     problem_name = curr_problem_row["name"][0]

    assert config.dataset_src == "local", "Only local dataset is supported for ThunderKitten"
    if config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

        

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    

    # For now: let's hardcode and use the toy problem as an example
    # ref_arch_src = read_file(os.path.join(REPO_TOP_DIR, "src/tk_prompts/toy_problem.py"))
    # problem_name = "TOY SUB PROBLEM"# toy problem
    
    # 2. Construct Prompt
    # TODO: @Simran this is where I need your help!!
    # --
    # Here is some idea
    # could we reference simple idea: https://github.com/HazyResearch/ThunderKittens/tree/tk_gen/simple_kernels
    # I try to turn this into a pybind
    custom_tk_prompt = prompt_generate_custom_thunderkitten_from_prompt_template(ref_arch_src)
    if config.log_prompt:
        with open(os.path.join(config.logdir, f"prompt_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
            f.write(custom_tk_prompt)

 
    # 3. Generate Sample
    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens,
                                                        verbose=config.verbose, 
                                                        time_generation=True)


    
    if not config.skip_generation:
        # # Query server with constructed prompt
        inference_result = inference_server(custom_tk_prompt)

        # here we will need to extract the cu (TK code kernel code) and the module with modified custom code
        kernel_code = extract_code_blocks_of_type(inference_result, "cpp")
        new_model_code = extract_code_blocks_of_type(inference_result, "python")

        
        assert kernel_code is not None, "Custom TK kernel generation failed"
        assert new_model_code is not None, "Custom Model code generation failed"
        
        if config.log:
            with open(os.path.join(config.logdir, f"inference_result_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
                f.write(inference_result)
            with open(os.path.join(config.logdir, f"kernel_code_level_{config.level}_problem_{config.problem_id}.cu"), "w") as f:
                f.write(kernel_code)
            with open(os.path.join(config.logdir, f"new_model_code_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
                f.write(new_model_code)


    # just for Debug, do not go pass here to build the kernel
    if config.stop_before_eval:
        return
    
    # 4. Build Kernel (TK Specific)
    kernel_dir = os.path.join(config.kernel_builds_dir, f"level_{config.level}_problem_{config.problem_id}")

    if os.path.exists(kernel_dir):
        print(f"Warning: Kernel directory {kernel_dir} already exists. Will overwrite contents.")
        # shutil.rmtree(kernel_dir)
    os.makedirs(kernel_dir, exist_ok=True)

    if not config.skip_generation:
        # Inside the directory (kernel_dir), we should have 3 files, the kernel module would be named tk_kernels
        # 1. custom_tk.cu: the cuda kernel itself (extracted from LLM response)
        with open(os.path.join(kernel_dir, "custom_tk.cu"), "w") as f:
            f.write(kernel_code)
    
    # 2. Makefile: the makefile to build the kernel (fixed for now)
    create_tk_makefile(kernel_dir)

    # 3. .so binary which would only be there if we succesfully build the kerne

    if config.clean_kernel_build:
        # Clean any existing .so files in kernel directory
        for file in os.listdir(kernel_dir):
            if file.endswith('.so'):
                so_path = os.path.join(kernel_dir, file)
                print(f"Removing existing built kernel binary: {so_path}")
                os.remove(so_path)

    # Run make inside the kernel directory to build the Kernel
    try:
        make_process = subprocess.run(
            ["make"], 
            cwd=kernel_dir,
            capture_output=True,
            text=True,
            check=True
        )
        print("Make output:", make_process.stdout)
    except subprocess.CalledProcessError as e:
        print("Make failed with error:", e.stderr)

        kernel_exec_result = KernelExecResult(
            compiled=False,
            correctness=False,
            runtime=-1.0,
            runtime_stats={},
            metadata={"error": str(e.stderr)}
        )
        raise RuntimeError("Failed to build kernel")
    
    # Load in compiled custom kernel module
    sys.path.append(kernel_dir) # point to the specific binary in the directory
    try:
        import tk_kernels # we name all the thunderkitten kernel modules as tk_kernels now!
        print(f"Imported ThunderKittens Kernel modules at {kernel_dir}: {dir(tk_kernels)}")
    except ImportError as e:
        print(f"Failed to import tk_kernels: {e}")
        kernel_exec_result = create_compilation_file_kernel_exec_result(f"Failed to import tk_kernels: {e}")

    # If we make it here, we have compiled and built the TK kernel!


    # 5. Evaluate the kernel, against original reference
    # TODO: @Simon need to write a TK specific eval function
    kernel_exec_result = None

    # # NOTE: no need to wrap around process here as only a single sample
    # # see batch eval for examples of process isolation

    if config.skip_generation:
        new_model_code = read_file(os.path.join(config.logdir, f"new_model_code_level_{config.level}_problem_{config.problem_id}.py"))  

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

