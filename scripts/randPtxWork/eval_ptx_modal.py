import json
import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modal
import pydra
from pydra import Config, REQUIRED
from src.utils import read_file

ptx_launcher_module = None  # global handle to the compiled extension

# Define the root path of your project *locally*
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- 1. Define the Modal App and Image (Corrected for very old client) ---
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "gcc-11", "g++-11", "clang")
    .env({"CC": "gcc-11", "CXX": "g++-11"})
    .pip_install(  # required to build flash-attn
        "anthropic",
        "numpy",
        "openai",
        "packaging",
        "pydra_config",
        "torch==2.5.0",
        "tqdm",
        "datasets",
        "transformers",
        "google-generativeai",
        "together",
        "pytest",
        "ninja",
        "utils",
        "python-dotenv",
    )
    # MODIFICATION IS HERE: Use `copy=True` instead of `copy_files=True`
    .add_local_dir(local_path=PROJECT_ROOT_DIR, remote_path="/project", copy=True)
    .run_commands(
        "cd /project && pip install -e .",
    )
)

app = modal.App("kernelbench-ptx-eval-final", image=image)

# --- 2. Define the Pydra Config Class ---
class EvalPtxConfig(Config):
    def __init__(self):
        self.ptx_file = REQUIRED
        self.json_file = REQUIRED
        self.level = REQUIRED
        self.problem_id = REQUIRED

# --- 3. Define the Remote Evaluation Class ---
@app.cls(gpu="A10G")
class PTXEvaluator:
    @modal.method()
    def evaluate(self, ref_code: str, ptx_code: str, ptx_meta): # Type hint removed
        """This method runs on the remote GPU."""
        import torch
        from src.eval import eval_kernel_against_ref
        import os
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        print("--- Running Evaluation on Modal GPU ---")
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

        custom_src_dict = {'ptx_code': ptx_code, 'ptx_meta': ptx_meta}
        result = eval_kernel_against_ref(
            original_model_src=ref_code,
            custom_model_src=custom_src_dict,
            device=torch.device("cuda:0"),
            backend='ptx',
            verbose=True,
            measure_performance=True
        )
        print("--- Evaluation on GPU Complete ---")
        if result:
            return result.model_dump()
        return None

# --- 4. Define the Main Logic ---
@pydra.main(base=EvalPtxConfig)
def main(config: EvalPtxConfig):
    # ... (This part is correct and does not need changes)
    print("--- Running Locally: Reading files ---")
    try:
        ptx_code = read_file(config.ptx_file)
        with open(config.json_file, 'r') as f:
            ptx_meta = json.load(f)

        from src.dataset import construct_kernelbench_dataset
        dataset_paths = construct_kernelbench_dataset(config.level)
        ref_problem_path = dataset_paths[config.problem_id - 1]
        ref_code = read_file(ref_problem_path)

        print(f"Loaded PTX file: {config.ptx_file}")
        print(f"Loaded JSON file: {config.json_file}")
        print(f"Loaded reference problem: {os.path.basename(ref_problem_path)}")

    except Exception as e:
        print(f"Error reading files locally: {e}")
        sys.exit(1)
        
    with app.run():
        print("\n--- Submitting job to Modal... ---")
        evaluator = PTXEvaluator()
        result_dict = evaluator.evaluate.remote(
            ref_code=ref_code,
            ptx_code=ptx_code,
            ptx_meta=ptx_meta,
        )

        print("\n--- MODAL JOB COMPLETE ---")
        if result_dict:
            print(f"  Compiled: {result_dict.get('compiled')}")
            print(f"  Correctness: {'PASS' if result_dict.get('correctness') else 'FAIL'}")
            print(f"  Performance (Runtime ms): {result_dict.get('runtime', 'N/A')}")
            print("\n  Metadata:")
            metadata = result_dict.get('metadata', {})
            print(json.dumps(metadata, indent=2))
        else:
            print("Evaluation did not return a result. Check Modal logs for errors.")

# --- 5. Standard Python entrypoint to kick off Pydra ---
if __name__ == "__main__":
    main()