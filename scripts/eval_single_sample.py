import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets

"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

def main():
    """
    Keep it simple: Generate and evaluate a single sample
    """
    all_problems = json.load(open(f"{REPO_TOP_DIR}/h100_xsm.json"))
    level = "1"
    task_id = "1"
    all_level = all_problems[level]
    task = next((item for item in all_level if str(item.get("task_id")) == str(task_id)), None)
    ref = task["ref_code"]
    src = task["custom_code"]
    
    print(f"Src:\n{src}")

    kernel_exec_result = eval_kernel_against_ref(
        ref, src, verbose=True, measure_performance=True, num_correct_trials=5, num_perf_trials=100
    )
    print(f"Evaluation result:\n{kernel_exec_result}")


if __name__ == "__main__":
    main()