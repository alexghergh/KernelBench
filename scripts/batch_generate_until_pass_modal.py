#!/usr/bin/env python3
"""batch_generate_until_pass_modal.py – Generate Triton kernels with an LLM
(facebook/KernelLLm by default) for a whole KernelBench level, retry until the
kernel passes correctness, and evaluate on a Modal GPU.

The workflow is completely self-contained:
  • prompt_generate_custom_triton_from_prompt_template -> build prompt
  • create_inference_server_from_presets(server_type="hf", model_name="facebook/KernelLLm") -> get code
  • modal_gpu_eval() (runs on GPU) -> compile & evaluate with src.eval
  • When correctness=True, save to  runs/<run_name>/level_<L>/<problem_name>/
      ├─ kernel.py
      └─ result.json

Usage (from repo root):
    modal run KernelBench/scripts/batch_generate_until_pass_modal.py \
        --level 1            # which level to process (1-4)
        --run_name kernelllm  # output directory name under runs/
        --max_tries 4        # how many attempts per problem

Environment prerequisites (local side):
    pip install modal datasets

The GPU side image already installs torch 2.2 + triton 2.2.
"""
from __future__ import annotations

import argparse, json, os, pathlib, time
from typing import Optional

import modal
from datasets import load_dataset

# ── Add repo root to PYTHONPATH for local execution ────────────────────── #
THIS_FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
import sys
# Ensure both repo root and the inner KernelBench/ are importable
paths_to_add = [PROJECT_ROOT, PROJECT_ROOT / "KernelBench"]
for p in paths_to_add:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

# ───────────────────────── Repo helper imports ────────────────────────── #
# ---------------------------------------------------------------------------
# Import prompt_constructor_triton even though the file has no ".py" suffix.
# ---------------------------------------------------------------------------
try:
    from src.prompt_constructor_triton import (
        prompt_generate_custom_triton_from_prompt_template,
    )
except ModuleNotFoundError:  # fallback – load from exact file path
    import importlib.util

    _pc_path = PROJECT_ROOT / "KernelBench" / "src" / "prompt_constructor_triton"
    if not _pc_path.exists():
        raise  # re-raise original error – file genuinely missing

    import importlib.machinery

    loader = importlib.machinery.SourceFileLoader(
        "src.prompt_constructor_triton", str(_pc_path)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    _module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    loader.exec_module(_module)  # type: ignore[attr-defined]

    sys.modules["src.prompt_constructor_triton"] = _module  # make import-able elsewhere
    prompt_generate_custom_triton_from_prompt_template = (
        _module.prompt_generate_custom_triton_from_prompt_template
    )
# note: extract_first_code is still needed
from src.utils import (
    extract_first_code,
    set_gpu_arch,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # /…/kb_research
RUNS_DIR  = REPO_ROOT / "runs"

################################################################################
# 1. Modal objects – GPU evaluation
################################################################################
CUDA_TAG = "12.1"  # keep <= host driver
# ---------------------------------------------------------------------------
# Base image: include transformers stack so the LLM can run inside the worker
# ---------------------------------------------------------------------------
BASE_IMAGE = (
    modal.Image
    .from_registry(f"pytorch/pytorch:2.2.2-cuda{CUDA_TAG}-cudnn8-runtime")
    .apt_install("git", "g++", "clang")
    .pip_install(
        "triton==2.2.0",
        "transformers>=4.40",
        "accelerate",
        "safetensors",
        "datasets",
        "sentencepiece",
        "bitsandbytes; platform_system=='Linux'",  # optional – ignored on macOS/Win
    )
    .env({"PYTHONPATH": "/workspace"})
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/workspace",
        ignore=["ptx_out/**", "**/__pycache__/**", ".git/**"],
    )
)

# ---------------------------------------------------------------------------
# Modal worker class – loads KernelLLm once, then generates & evaluates.
# ---------------------------------------------------------------------------

app = modal.App("kb-batch-eval")

@app.cls(gpu="A10G", image=BASE_IMAGE, timeout=60*30, memory=24_576)
class KBWorker:
    def __enter__(self):
        # Lazy import to keep host env light
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("[Worker] Loading facebook/KernelLLm …", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/KernelLLm", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "facebook/KernelLLm",
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    @modal.method()
    def generate_and_eval(self, ref_src: str, max_tokens: int = 2048, temperature: float = 0.0):
        # local imports inside container
        from src.prompt_constructor_triton import (
            prompt_generate_custom_triton_from_prompt_template,
        )
        from src.eval import eval_kernel_against_ref

        prompt = prompt_generate_custom_triton_from_prompt_template(ref_src)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(max_new_tokens=max_tokens, temperature=temperature)
        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        output = self.model.generate(**inputs, **gen_kwargs)
        raw_text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        from src.utils import extract_first_code as _efc
        code_block = _efc(raw_text, ["python"])
        if not code_block:
            return None, None  # signal failure

        # Compile & evaluate
        from src.utils import set_gpu_arch as _set_arch
        _set_arch(["Ada"])
        result = eval_kernel_against_ref(
            original_model_src=ref_src,
            custom_model_src=code_block,
            backend="triton",
            num_correct_trials=5,
            num_perf_trials=100,
            measure_performance=True,
            verbose=False,
        )
        return code_block, result

################################################################################
# 2. Driver logic – runs locally
################################################################################

def process_level(
    level: int,
    model_name: str,
    max_tries: int,
    run_name: str,
    dataset_src: str = "huggingface",
):
    # ----- load dataset -----
    if dataset_src == "huggingface":
        ds = load_dataset("ScalingIntelligence/KernelBench")[f"level_{level}"]
    else:
        raise NotImplementedError("Only huggingface dataset supported in this script")

    out_root = RUNS_DIR / run_name / f"level_{level}"
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- local LLM inference wrapper -----
    # llm = create_inference_server_from_presets(
    #     server_type="hf",
    #     model_name=model_name,
    #     max_tokens=2048,
    #     temperature=0.0,
    # )

    print(f"[Info] Processing Level {level} – {len(ds)} problems")

    for row in ds:
        prob_id   : int  = row["problem_id"]
        prob_name : str  = row["name"]  # e.g., "23_Softmax.py"
        ref_src   : str  = row["code"]

        save_dir = out_root / prob_name
        save_dir.mkdir(parents=True, exist_ok=True)
        kernel_path = save_dir / "kernel.py"
        result_path = save_dir / "result.json"

        if kernel_path.exists():
            print(f"[skip] {prob_name} already solved -> {kernel_path}")
            continue

        for attempt in range(1, max_tries + 1):
            print(f"[{prob_id}] attempt {attempt} – querying LLM …", end="", flush=True)
            # prompt = prompt_generate_custom_triton_from_prompt_template(ref_src)
            # llm_raw = llm(prompt)
            # triton_code = extract_first_code(llm_raw, ["python"])
            # if not triton_code:
            #     print(" no code-block found → retry")
            #     continue

            # ---------- remote GPU evaluation ----------
            try:
                # result = modal_gpu_eval.remote(ref_src, triton_code, gpu_arch=["Ada"])
                with KBWorker() as worker:
                    triton_code, result = worker.generate_and_eval(ref_src)
            except Exception as exc:
                print(f" Modal error: {exc} → retry")
                continue

            if result and getattr(result, "correctness", False):
                print(" ✓ passes")
                kernel_path.write_text(triton_code)
                result_path.write_text(json.dumps(result.model_dump(), indent=2))
                break
            else:
                print(" × fails")
        else:
            print(f"[FAIL] {prob_name} – all {max_tries} tries exhausted")

# ---------------------------------------------------------------------------
# 3. Entry-point for `modal run …` (Typer will map CLI flags -> args)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(level: int, run_name: str, max_tries: int = 3, model_name: str = "facebook/KernelLLm"):
    """Local entry-point invoked by `modal run`.

    Example:
        modal run KernelBench/scripts/batch_generate_until_pass_modal.py \
            --level 1 --run-name kernelllm_l1 --max-tries 4
    """
    process_level(
        level=level,
        model_name=model_name,
        max_tries=max_tries,
        run_name=run_name,
    )

# Allow running the script directly with `python …` as well
if __name__ == "__main__":
    import sys
    if "modal" in sys.argv[0]:
        # Running via `modal run`, the local entrypoint will be used
        pass
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--level", type=int, required=True)
        parser.add_argument("--run_name", type=str, required=True)
        parser.add_argument("--max_tries", type=int, default=3)
        parser.add_argument("--model_name", type=str, default="facebook/KernelLLm")
        cfg = parser.parse_args()
        process_level(cfg.level, cfg.model_name, cfg.max_tries, cfg.run_name) 