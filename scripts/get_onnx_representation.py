import os
os.environ["TORCH_LOGS_OUT"] = "torch_compile_output.log"
import onnx
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
Get ONNX Representation of Model

You need to pip install onnx onnxruntime

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

def get_onnx_representation(
        ref_arch_name: str,
        ref_arch_src: str, 
        output_path: str,
        device: torch.device="cuda:0",
        verbose: bool = False,
) -> None:
    """
    Get ONNX Representation of Model
    """
    print(f"Getting ONNX representation of {ref_arch_name}")

    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )

    try:
        with torch.no_grad():
            # Setup inputs
            set_seed(42)
            inputs = get_inputs()
            init_inputs = get_init_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            # Initialize model
            model = Model(*init_inputs)
            model = model.cuda(device=device)
            
            # Export to ONNX
            # see guide at https://pytorch.org/docs/stable/onnx.html
            torch.onnx.export(
                model,                  # model to export
                tuple(inputs),          # inputs of the model,
                output_path,            # filename of the ONNX model
                input_names=["input"],  # Rename inputs for the ONNX model
                dynamo=True             # True or False to select the exporter to use
            )
            
            if verbose:
                print(f"Saved ONNX model to {output_path}")
            
    except Exception as e:
        print(f"[Eval] Error in ONNX Export: {e}")


def visualize_onnx(onnx_path: str):
    """
    Load and visualize ONNX model in different formats
    """
    # Load the ONNX model
    model = onnx.load(onnx_path)
    
    print("\n1. Text Format (Proto):")
    print(model)
    
    print("\n2. Graph Structure:")
    graph = model.graph
    print(f"Inputs: {[input.name for input in graph.input]}")
    print(f"Outputs: {[output.name for output in graph.output]}")
    print("\nNodes:")
    for node in graph.node:
        print(f"Op Type: {node.op_type}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")
        print(f"  Attributes: {[attr.name for attr in node.attribute]}")
        print()

    print("\n3. JSON Format:")
    # Convert to JSON (need to handle some data types)
    model_dict = {
        "nodes": [{
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": [attr.name for attr in node.attribute]
        } for node in graph.node],
        "inputs": [input.name for input in graph.input],
        "outputs": [output.name for output in graph.output]
    }
    print(json.dumps(model_dict, indent=2))


if __name__ == "__main__":
    """
    Get the ONNX representation of a particular model
    """


    device = torch.device("cuda:0")

    # from add example
    ref_arch_path = os.path.join(REPO_TOP_PATH, "src", "prompts", "representations", "model_ex_pytorch.py")
    ref_arch_name = "model_ex_pytorch"
    ref_arch_src = read_file(ref_arch_path)


    # from KernelBench
    # level_num = 2

    # problem_id = 80
    # PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level_num))
    # dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)

    # ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)


    # Create output directory if it doesn't exist
    output_dir = os.path.join(REPO_TOP_PATH, "results", "representations")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{ref_arch_name}.onnx")

    get_onnx_representation(
        ref_arch_name=ref_arch_name,
        ref_arch_src=ref_arch_src,
        output_path=output_path,
        device=device,
        verbose=True)
    
    visualize_onnx(output_path)
    
    

    