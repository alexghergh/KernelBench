from src.utils import read_file, set_gpu_arch
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
import os
import logging
import json
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda:0")

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def fetch_ref_arch_from_level_problem_id(level_num, problem_id, with_name=False):
    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level_num))
    dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
    return fetch_ref_arch_from_problem_id(problem_id, dataset, with_name)

def run_profile(level_num, problem_id, num_trials=10):
    ref_arch_name, ref_arch_src = fetch_ref_arch_from_level_problem_id(
        level_num, problem_id, with_name=True
    )
    ref_arch_name = ref_arch_name.split("/")[-1]
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    # try:
    with torch.no_grad():
        profiling_scheduler = torch.profiler.schedule(
            wait=1,
            warmup=2,
            active=7,
        )
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
        
        # Create base model
        model = Model(*init_inputs)
        model = model.cuda(device=device)
        
        # Profile non-compiled model
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=profiling_scheduler
        ) as prof:
            with record_function("non_compiled_forward"):
                for _ in range(num_trials):
                    model(*inputs)
                    prof.step()
        print(f"\nProfiling results for non-compiled model:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # Profile compiled model
        model_compiled = torch.compile(model)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof_compiled:
            with record_function("compiled_forward"):
                for _ in range(num_trials):
                    model_compiled(*inputs)
                    prof_compiled.step()
        print(f"\nProfiling results for compiled model:")
        print(prof_compiled.key_averages().table(sort_by="cuda_time_total", row_limit=10))


        prof.export_chrome_trace("trace_non_compiled.json")
        prof_compiled.export_chrome_trace("trace_compiled.json")

    # except Exception as e:
        # print(f"[Eval] Error in Measuring Performance: {e}")


def get_time_tk(problem_id, num_trials=100, torch_compile=False, torch_compile_mode: str = None):
    """
    From TK directory
    """
    TK_DIR = os.path.join(KERNEL_BENCH_PATH, "tk")
    dataset_level_tk = construct_problem_dataset_from_problem_dir(TK_DIR)

    ref_arch_path = None
    for problem_path in dataset_level_tk:
        problem_name = os.path.basename(problem_path)
        problem_number = int(problem_name.split("_")[0])
        if problem_number == problem_id:
            ref_arch_path = problem_path
            break
    assert ref_arch_path is not None, f"Problem {problem_id} not found in dataset"
    problem_name = os.path.basename(ref_arch_path)
    ref_arch_name = problem_name.split("/")[-1]
    ref_arch_src = read_file(ref_arch_path)
    
    # import pdb; pdb.set_trace()
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
            model = Model(*init_inputs)
            if torch_compile:
                if torch_compile_mode:
                    model = torch.compile(model, mode=torch_compile_mode)
                else:
                    model = torch.compile(model)
            model = model.cuda(device=device)
            torch.cuda.synchronize(device=device)
            elapsed_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=num_trials, verbose=False, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)
            # json_results[f"level{level_num}"][ref_arch_name] = runtime_stats
            print(f"{ref_arch_name} {runtime_stats}")
            return (ref_arch_name, runtime_stats)
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")
        return (None, None)


# def get_torch_compile_triton(level_num, problem_id, mode=None):
#     """
#     Get the triton code generated by torch compile for a particular problem
#     """
#     ref_arch_name, ref_arch_src = fetch_ref_arch_from_level_problem_id(
#         level_num, problem_id, with_name=True
#     )
#     ref_arch_name = ref_arch_name.split("/")[-1]
#     context = {}
#     Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
#         ref_arch_src, context
#     )
#     try:
#         with torch.no_grad():
#             torch.cuda.synchronize(device=device)
#             set_seed(42)
#             inputs = get_inputs()
#             set_seed(42)
#             init_inputs = get_init_inputs()
#             inputs = [
#                 x.cuda(device=device) if isinstance(x, torch.Tensor) else x
#                 for x in inputs
#             ]
#             init_inputs = [
#                 x.cuda(device=device) if isinstance(x, torch.Tensor) else x
#                 for x in init_inputs
#             ]
#             model = Model(*init_inputs)

#             # output triton code
#             log_file = f"results/triton_code/level{level_num}_problem{problem_id}_triton.log"
#             # os.makedirs(os.path.dirname(log_file), exist_ok=True)
#             # logging.basicConfig(filename=log_file, level=logging.INFO)
#             # TODO: Figure out a way to save to a file 

#             torch._logging.set_logs(output_code=True)
#             # Call torch compile
#             if mode:
#                 model = torch.compile(model, mode=mode)
#             else:
#                 model = torch.compile(model)
#             model = model.cuda(device=device)
#             torch.cuda.synchronize(device=device)
#             elapsed_times = time_execution_with_cuda_event(
#                 model, *inputs, num_trials=1, verbose=False, device=device
#             )
#             # runtime_stats = get_timing_stats(elapsed_times, device=device)
#             # json_results[f"level{level_num}"][ref_arch_name] = runtime_stats
#             # print(f"{ref_arch_name} {runtime_stats}")
#             return (ref_arch_name)
#     except Exception as e:
#         print(f"[Eval] Error in Measuring Performance: {e}")


def record_baseline_times():
    """
    Record the baseline times for all problems in all levels
    Record 3 things
    1. torch_compile: off
    2. torch_compile: on with no flag
    3. torch_compile: on + mode set to max-autotune
    """


    record_settings = [
        (False, None),
        (True, None),
        (True, "max-autotune"),
    ]
    
    for torch_compile, torch_compile_mode in record_settings:
        # whether to use torch compile or not, or which mode to use
        json_results = {}

        PROBLEM_DIR_TK = "KernelBench/tk"
        dataset_level_tk = construct_problem_dataset_from_problem_dir(PROBLEM_DIR_TK)
        json_results["level_tk"] = {}

        problem_ids = [int(os.path.basename(problem_path).split("_")[0]) for problem_path in dataset_level_tk]
    
        for problem_id in problem_ids:
            ref_arch_name, runtime_stats = get_time_tk(
                problem_id, num_trials=100, torch_compile=torch_compile, torch_compile_mode=torch_compile_mode
            )
            json_results["level_tk"][ref_arch_name] = runtime_stats

        if not torch_compile:
            save_path = f"results/timing/tk/baseline_time.json"
        elif torch_compile:
            if torch_compile_mode:
                save_path = f"results/timing/tk/baseline_time_torch_compile_mode_{torch_compile_mode}.json"
            else:
                save_path = f"results/timing/tk/baseline_time_torch_compile.json"


        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(json_results, f)

if __name__ == "__main__":

    set_gpu_arch(["Hopper"])
    
    record_baseline_times()
    # get_torch_compile_triton(1, 12)
    # record_baseline_times()

    # run_profile(2, 43)
    # get_time(2, 43, torch_compile=False)
    # get_time(2, 43, torch_compile=True)
