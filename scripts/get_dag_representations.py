import os
os.environ["TORCH_LOGS_OUT"] = "torch_compile_output.log"

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

"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")


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


def get_torch_dag(
        ref_arch_name: str,
        ref_arch_src: str, 
        device: torch.device="cuda:0",
        verbose: bool = False,
) -> torch.fx.GraphModule.graph:
    """
    Get DAG Representation of Torch Compiled Model
    """

    print(f"Getting DAG representation of {ref_arch_name}")

    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )

    # set environment variables for torch compile logging
    torch._logging.set_logs(output_code=True)

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


            # We just need to torch nn.module to grab the DAG representation
            # See Torch FX documentation: https://pytorch.org/docs/stable/fx.html

            # Symbolic tracing frontend - captures the semantics of the module
            symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)

            # High-level intermediate representation (IR) - Graph representation
            dag_graph = symbolic_traced.graph

            # no need run it            

            # # just run it forward            
            # model = model.cuda(device=device)
            # torch.cuda.synchronize(device=device)

            # # import pdb; pdb.set_trace()

            # elapsed_times = time_execution_with_cuda_event(
            #     model, *inputs, num_trials=1, verbose=verbose, device=device
            # )
            # runtime_stats = get_timing_stats(elapsed_times, device=device)

            # if verbose:
            #     print(f"{ref_arch_name} {runtime_stats}")
            
            return dag_graph
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")


def test_get_dag_representation_particular_program(level_num: int, problem_id: int):
    """
    Get the DAG representation of the model
    """
    device = torch.device("cuda:0")

    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level_num))
    dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)

    ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)

    dag_graph = get_torch_dag(
        ref_arch_name=ref_arch_name,
        ref_arch_src=ref_arch_src,
        device=device,
        verbose=False)

    print(f"DAG Graph for {ref_arch_name}")
    print(dag_graph)

    print(f"DAG Graph for {ref_arch_name} in tabular format")
    print(dag_graph.print_tabular())

if __name__ == "__main__":
    # DEBUG and simple testing
    # test_measure_particular_program(2, 28)
    
    test_get_dag_representation_particular_program(level_num=2, problem_id=80)