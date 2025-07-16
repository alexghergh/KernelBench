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

    grid_x, block_x = _extract_launch_line(llm_out)
    print(f"[PTX] Using launch grid=({grid_x},1,1) block=({block_x},1,1)")

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
        kernel = compile_ptx(ptx_code, "my_kernel", grid=(grid_x, 1, 1), block=(block_x, 1, 1))

        # For now assume single-output tensor same shape as ref_out
        out = torch.empty_like(ref_out)
        # Move output/input to CuPy via DLPack
        kernel(out, *inputs)

        # out is mutated in-place on device; ensure torch sync
        torch.cuda.synchronize(device)

        correct = _compare(out, ref_out)
        print(json.dumps({"correct": correct}))


if __name__ == "__main__":
    main() 