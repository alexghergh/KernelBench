import os, argparse, json, pathlib

import modal

################################################################################
# Modal image – based on CUDA runtime + PyTorch + CuPy + OpenAI client
################################################################################
CUDA_VERSION = "12.4.0"
TAG = f"{CUDA_VERSION}-devel-ubuntu22.04"

REPO_TOP = pathlib.Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install(
        "torch==2.5.0",
        "cupy-cuda12x",  # generic CUDA-12 wheel (works for 12.4 runtime)
        "tqdm",
        "openai",
        "python-dotenv",
        "together",  # utils imports Together client
        "transformers",
        "google-generativeai",
        "anthropic",
        "numpy",
        "openai",  # ensure client in container as well
        "packaging",
        "pydra_config",
        "torch==2.5.0",
        "tqdm",
        "datasets",
        "transformers",
        "google-generativeai",
        "together",
        "anthropic",
        "pytest",
        "ninja",
        "utils",
        "einops",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(REPO_TOP / "KernelBench", remote_path="/root/KernelBench")
    .add_local_dir(REPO_TOP / "src", remote_path="/root/src")
)

app = modal.App("generate-eval-ptx-modal")


################################################################################
# Remote class that runs on GPU
################################################################################

@app.cls(gpu="H100", image=image, timeout=600, secrets=[modal.Secret.from_name("openai-key")])
class PTXEvaluator:
    @modal.method()
    def generate_and_eval(
        self,
        level: int,
        problem_id: int,
        server: str = "openai",
        model: str = "gpt-4o-mini",
        verbose: bool = False,
    ) -> dict:
        """Generate PTX via LLM and evaluate correctness. Runs on remote GPU."""
        import torch, os, json, re
        from pathlib import Path

        from src.dataset import construct_problem_dataset_from_problem_dir
        from src.utils import query_server, extract_first_code, load_dotenv
        from src.ptx_prompt_constructor import make_ptx_prompt
        from src.ptx_utils import compile_ptx
        from src.eval import load_original_model_and_inputs, set_seed

        # ensure .env key is loaded
        load_dotenv()

        repo_root = Path("/root")
        level_dir = repo_root / "KernelBench" / f"level{level}"
        dataset = construct_problem_dataset_from_problem_dir(str(level_dir))
        pattern = f"{problem_id}_"
        problem_path = next(p for p in dataset if Path(p).name.startswith(pattern))
        problem_src = Path(problem_path).read_text()

        prompt = make_ptx_prompt(problem_src)
        extra = {}
        if server == "openai" and ("o3" in model or "o1" in model):
            extra["is_reasoning_model"] = True
            extra["reasoning_effort"] = "high"

        llm_out = query_server(prompt,
                               server_type=server,
                               model_name=model,
                               max_tokens=1024,
                               temperature=0.0,
                               **extra)

        if verbose:
            print("===== RAW OUTPUT =====")
            print(llm_out)
            print("======================")

        # parse launch dims
        m = re.search(r"LAUNCH\s*=\s*\((\d+)\s*,\s*(\d+)\)", llm_out)
        grid_x, block_x = (int(m.group(1)), int(m.group(2))) if m else (80, 256)

        ptx_code = extract_first_code(llm_out, ["ptx"])
        if ptx_code is None:
            return {"error": "no_ptx_block"}

        # reference model
        set_seed(0)
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(problem_src, {})
        device = torch.device("cuda")
        with torch.no_grad():
            init_args = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_init_inputs()]
            ref_model = Model(*init_args).to(device)
            inp = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
            ref_out = ref_model(*inp)

            try:
                kernel = compile_ptx(ptx_code, "my_kernel", grid=(grid_x, 1, 1), block=(block_x, 1, 1))
                out = torch.empty_like(ref_out)
                kernel(out, *inp)
                torch.cuda.synchronize()
                correct = torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3)
                return {
                    "correct": correct,
                    "grid": grid_x,
                    "block": block_x,
                    "llm_out": llm_out if verbose else None,
                }
            except Exception as e:
                return {
                    "error": "compile_or_runtime",
                    "message": str(e),
                    "grid": grid_x,
                    "block": block_x,
                    "llm_out": llm_out if verbose else None,
                }

        # unreachable but for syntax
        return {"error": "unknown"}


################################################################################
# Convenience wrapper
################################################################################

def run_remote(level: int, problem_id: int, **kwargs):
    import modal  # local import to enable logging only here
    modal.enable_output()  # stream build / run logs to local console
    with app.run():
        return PTXEvaluator().generate_and_eval.remote(level, problem_id, **kwargs)


################################################################################
# CLI
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PTX and evaluate via Modal GPU")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--server", default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    result = run_remote(args.level, args.problem_id, server=args.server, model=args.model, verbose=args.verbose)
    print(json.dumps(result, indent=2)) 