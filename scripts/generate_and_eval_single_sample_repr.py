import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_alternative_representation
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets

"""
Generate and evaluate a single sample

This is for the ablation to test different representations of the problem

4 options
1. PyTorch
2. Natural language
3. ONNX graph of PyTorch program 
4. Torch FX representation of PyTorch program

Usage:
python3 scripts/generate_and_eval_single_sample_repr.py dataset_src=local representation=pytorch log_prompt=True

Right now I fixed it for level 2 problem 19
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

class EvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"


        # Problem Specification
        self.level = 2
        # NOTE: this is the logical index (problem id the problem_name)\
        self.problem_id = 19

        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "local"
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ["Ada"]

        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.0

        # Representation    
        # options: "pytorch", "nl", "onnx", "torch_fx"
        self.representation = REQUIRED


        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False

        self.log = False
        self.log_prompt = False
        self.log_generated_kernel = False
        self.log_eval_result = False


    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


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

    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)  # otherwise build for all architectures

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)
        
    # Problem Checks
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"


    # 1. Fetch Problem
    if config.dataset_src == "huggingface":

        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)
    # import pdb; pdb.set_trace()

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    

    assert config.level == 2, "Only level 2 is supported for now"   
    assert config.problem_id == 19, "Only testing this problem 19"
    
    # 2. Generate Sample
    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens,
                                                        verbose=config.verbose,
                                                        time_generation=True)
    
    print(f"Generating prompt for representation: {config.representation}")

    EX_PROMPT_DIR = os.path.join(REPO_TOP_DIR, "src", "prompts", "representations")
    PROBLEM_PROMPT_DIR = os.path.join(REPO_TOP_DIR, "19_ConvTranspose2d_GELU_GroupNorm")
    if config.representation == "pytorch":
        ref_arch_src = read_file(os.path.join(PROBLEM_PROMPT_DIR, "model_pytorch.py"))
        example_arch_src = read_file(os.path.join(EX_PROMPT_DIR, "model_ex_pytorch.py"))
    elif config.representation == "nl":
        ref_arch_src = read_file(os.path.join(PROBLEM_PROMPT_DIR, "model_nl.txt"))
        example_arch_src = read_file(os.path.join(EX_PROMPT_DIR, "model_ex_nl.txt"))
    elif config.representation == "onnx":
        ref_arch_src = read_file(os.path.join(PROBLEM_PROMPT_DIR, "model_onnx_graph.txt"))
        example_arch_src = read_file(os.path.join(EX_PROMPT_DIR, "model_ex_onnx_graph.txt"))
    elif config.representation == "torch_fx":
        ref_arch_src = read_file(os.path.join(PROBLEM_PROMPT_DIR, "model_torch_fx.txt"))
        example_arch_src = read_file(os.path.join(EX_PROMPT_DIR, "model_ex_torch_fx.txt"))
    
    example_new_arch_src = read_file(os.path.join(EX_PROMPT_DIR, "model_new_ex.py"))


    custom_cuda_prompt = prompt_generate_alternative_representation(ref_arch_src, 
                                                                    example_arch_src, 
                                                                    example_new_arch_src, 
                                                                    config.representation)
    if config.log_prompt:
        with open(os.path.join(config.logdir, f"prompt_level_{config.level}_problem_{config.problem_id}_repr_{config.representation}.txt"), "w") as f:
            f.write(custom_cuda_prompt)

    # Query server with constructed prompt
    custom_cuda = inference_server(custom_cuda_prompt)
    custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"
    
    # this should be optional
    if config.log:
        with open(os.path.join(config.logdir, f"generated_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_cuda)

    # 3. Evaluate Kernel
    # NOTE: no need to wrap around process here as only a single sample
    # see batch eval for examples of process isolation
    kernel_exec_result = eval_kernel_against_ref(
        ref_arch_src, custom_cuda, verbose=config.verbose, measure_performance=True, num_correct_trials=5, num_perf_trials=100
    )
    
    print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")

    if config.log:
        with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}_repr_{config.representation}.txt"), "a") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(str(kernel_exec_result))


if __name__ == "__main__":
    main()

