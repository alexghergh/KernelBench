#!/usr/bin/env python3
"""generate_and_eval_single_sample_ptx.py

Query an LLM for **PTX** code that implements a given KernelBench problem,
compile it with CuPy, and compare its output to the reference PyTorch model.

This is a minimal counterpart to generate_and_eval_single_sample.py but for PTX.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch

from src.eval import (
    load_original_model_and_inputs,
    set_seed,
)
from src.utils import query_server, extract_first_code
from src.ptx_prompt_constructor import make_ptx_prompt
from src.ptx_utils import compile_ptx
from src.dataset import construct_problem_dataset_from_problem_dir

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _extract_launch_line(text: str) -> tuple[int, int]:
    """Parse a line like ``LAUNCH = (80, 256)``.
    Returns (grid_x, block_x).  Defaults to (80, 256) if not found.
    """
    m = re.search(r"LAUNCH\s*=\s*\((\d+)\s*,\s*(\d+)\)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 80, 256


def _compare(a: torch.Tensor, b: torch.Tensor, atol=1e-3, rtol=1e-3):
    if a.shape != b.shape:
        return False
    return torch.allclose(a, b, atol=atol, rtol=rtol)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate PTX with an LLM and evaluate it.")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--server", default="openai", help="LLM provider type for utils.query_server (openai, together, deepseek, etc.)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name for the chosen provider")
    parser.add_argument("--trials", type=int, default=10, help="Metric collection trials (unused for now)")
    parser.add_argument("--device", default="cuda", help="Torch device for reference execution")
    parser.add_argument("--verbose", action="store_true", help="Print raw LLM output and intermediate info")
    parser.add_argument("--output-root", default="ptx_test_results", help="Root directory to save run artefacts (default: ptx_out)")

    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # 1. Locate problem
    # ---------------------------------------------------------------------
    repo_root = Path(__file__).resolve().parent.parent
    level_dir = repo_root / "KernelBench" / f"level{args.level}"
    dataset = construct_problem_dataset_from_problem_dir(str(level_dir))

    pattern = f"{args.problem_id}_"
    problem_path = None
    for p in dataset:
        if Path(p).name.startswith(pattern):
            problem_path = p
            break
    if problem_path is None:
        raise FileNotFoundError(f"Problem {args.problem_id} not found in level {args.level}")

    problem_src = Path(problem_path).read_text()

    # ---------------------------------------------------------------------
    # 2. Prompt LLM
    # ---------------------------------------------------------------------
    prompt = make_ptx_prompt(problem_src)
    llm_out = query_server(prompt,
                           server_type=args.server,
                           model_name=args.model,
                           max_tokens=1024,
                           temperature=0.0)

    if args.verbose:
        print("========== RAW LLM OUTPUT ==========")
        print(llm_out)
        print("====================================")

    ptx_code = extract_first_code(llm_out, ["ptx"])
    if ptx_code is None:
        raise RuntimeError("LLM output did not contain a PTX code block")

# ---------------------------------------------------------------------
# 2b.  Prepare output directory
# ---------------------------------------------------------------------
    # Use model name (args.model) to categorise LLM runs
    # Problem folder based on problem file stem (e.g. '3_Batched_matrix_multiplication')
    problem_stem = Path(problem_path).stem  # without .py
    out_dir = Path(args.output_root) / args.model / problem_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive unique run directory by timestamp
    # Use sequential numbering: run_001, run_002, ...
    existing_runs = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    next_idx = 1
    if existing_runs:
        # extract numeric parts and find max
        ids = []
        for p in existing_runs:
            try:
                ids.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        if ids:
            next_idx = max(ids) + 1
    run_dir = out_dir / f"run_{next_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ptx_path = run_dir / "kernel.ptx"
    summary_path = run_dir / "run.json"

    # Persist PTX code (always useful)
    try:
        ptx_path.write_text(ptx_code)
    except Exception as e:
        print(f"[WARN] Failed to write PTX file to {ptx_path}: {e}")

    # ---------------------------------------------------------------------
    # 3. Load reference model
    # ---------------------------------------------------------------------
    set_seed(0)
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(problem_src, {})

    device = torch.device(args.device)

    with torch.no_grad():
        set_seed(0)
        init_args = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_init_inputs()]
        ref_model = Model(*init_args).to(device)

        set_seed(0)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
        ref_out = ref_model(*inputs)

        # -----------------------------------------------------------------
        # 4. Compile & run PTX
        # -----------------------------------------------------------------
        grid_x, block_x = _extract_launch_line(ptx_code)
        kernel = compile_ptx(ptx_code, "my_kernel", grid=(grid_x, 1, 1), block=(block_x, 1, 1))

        # For now assume single-output tensor same shape as ref_out
        out = torch.empty_like(ref_out)
        # Move output/input to CuPy via DLPack
        kernel(out, *inputs)

        # out is mutated in-place on device; ensure torch sync
        torch.cuda.synchronize(device)

        correct = _compare(out, ref_out)
        result_obj = {
            "level": args.level,
            "problem_id": args.problem_id,
            "model": args.model,
            "grid_x": grid_x,
            "block_x": block_x,
            "correct": correct,
        }

        # Combine LLM raw output with result into one JSON
        run_summary = {
            "llm_raw": llm_out,
            "result": result_obj,
        }

        try:
            summary_path.write_text(json.dumps(run_summary, indent=2))
        except Exception as e:
            print(f"[WARN] Failed to write summary to {summary_path}: {e}")

        print(json.dumps(run_summary))


if __name__ == "__main__":
    main() 