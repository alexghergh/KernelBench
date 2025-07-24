import os, argparse, json, pathlib

import modal

################################################################################
# Modal image – based on CUDA runtime + PyTorch + CuPy + OpenAI client
################################################################################
CUDA_VERSION = "12.4.0"
TAG = f"{CUDA_VERSION}-devel-ubuntu22.04"

# Path to repository root (three levels up: PtxWork -> scripts -> KernelBench -> <repo root>)
# Robust path handling – if the script is copied to /root inside the container
# the parents chain may be shorter.  Fallback to /root when out of range.
_resolved = pathlib.Path(__file__).resolve()
parents = _resolved.parents
if len(parents) >= 4:
    REPO_ROOT = parents[3]
else:
    REPO_ROOT = pathlib.Path("/root")

# Convenience paths
KERNEL_BENCH_PATH = REPO_ROOT / "KernelBench"
SRC_PATH = REPO_ROOT / "src"
# Fallback – some repo layouts keep src/ inside KernelBench/
if not SRC_PATH.is_dir():
    fallback = REPO_ROOT / "KernelBench" / "src"
    if fallback.is_dir():
        SRC_PATH = fallback
    else:
        raise FileNotFoundError(f"Cannot locate src directory at {SRC_PATH} or {fallback}")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install(
        "torch==2.5.0",
        "cupy-cuda12x",
        "tqdm",
        "openai",
        "python-dotenv",
        "together",
        "transformers",
        "google-generativeai",
        "anthropic",
        "numpy",
        "packaging",
        "pydra_config",
        "datasets",
        "pytest",
        "ninja",
        "utils",
        "einops",
        "accelerate",
        "safetensors",
        "bitsandbytes",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(KERNEL_BENCH_PATH, remote_path="/root/KernelBench")
    .add_local_dir(SRC_PATH, remote_path="/root/src")
)

app = modal.App("generate-eval-ptx-modalv3")


################################################################################
# Remote class that runs on GPU
################################################################################

@app.cls(
    gpu="H100",
    image=image,
    timeout=600,
    retries=modal.Retries(max_retries=0),  # disable automatic retry; single shot per problem
    secrets=[
        modal.Secret.from_name("openai-key"),
        modal.Secret.from_name("anthropic-key"),  # for Claude/Anthropic
    ],
)
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
        # ---- Robust level directory resolution --------------------------------------
        # Depending on how the repo is copied into the Modal image, the KernelBench
        # problems may live in one of several locations.  We therefore probe a list of
        # candidate directories and pick the first one that exists.  This avoids the
        # intermittent "No such file or directory: /root/KernelBench/levelX" errors.

        level_dir_candidates = [
            repo_root / "KernelBench" / f"level{level}",            # /root/KernelBench/level1
            repo_root / "KernelBench" / "KernelBench" / f"level{level}",  # /root/KernelBench/KernelBench/level1
            repo_root / f"level{level}",                               # /root/level1 (rare)
        ]

        level_dir = next((p for p in level_dir_candidates if p.is_dir()), None)
        if level_dir is None:
            raise FileNotFoundError(
                f"Cannot locate level directory for level={level}. Tried: "
                + ", ".join(str(p) for p in level_dir_candidates)
            )

        if verbose:
            print(f"[DEBUG] Using level_dir: {level_dir}")

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
                # Pre-fill output with NaNs and pass it last so arg order matches (input_ptr, output_ptr)
                out = torch.full_like(ref_out, float('nan'))
                kernel(*inp, out)
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