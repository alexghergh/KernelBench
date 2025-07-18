#!/usr/bin/env python3
"""generate_and_eval_modal_ptx_batch.py
Run PTX generation + evaluation for a list of KernelBench problems via Modal.
Each invocation is archived under:
    ptx_test_results/<model>/<problem_stem>/run_###/
containing run.json (LLM raw + result) and kernel.ptx (generated CUDA/PTX).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime

# -----------------------------------------------------------------------------
# Repository paths & dynamic import setup
# -----------------------------------------------------------------------------
# This file lives in   <repo>/KernelBench/scripts/PtxWork/
# We want REPO_ROOT → <repo>/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Make sibling helper module importable
PTX_WORK_DIR = REPO_ROOT / "KernelBench" / "scripts" / "PtxWork"
sys.path.insert(0, str(PTX_WORK_DIR))

from generate_and_eval_single_sample_modal_ptx import run_remote  # noqa: E402
from src.dataset import construct_problem_dataset_from_problem_dir  # noqa: E402

# Where to store artefacts
OUT_DIR = REPO_ROOT / "ptx_test_results"
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Batch PTX generation/eval via Modal")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--problem-ids", type=str, help="Comma-separated list, e.g. 1,5,7")
    p.add_argument("--all", action="store_true", help="Run every problem in the level")
    p.add_argument("--server", default="openai")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Dataset path (note double KernelBench directory in repo layout)
    level_dir = REPO_ROOT / "KernelBench" / "KernelBench" / f"level{args.level}"
    if not level_dir.is_dir():
        raise FileNotFoundError(f"Level directory not found: {level_dir}")

    dataset = construct_problem_dataset_from_problem_dir(str(level_dir))

    if args.all:
        problem_ids = [int(pathlib.Path(p).name.split("_")[0]) for p in dataset]
    elif args.problem_ids:
        problem_ids = [int(tok) for tok in args.problem_ids.split(',') if tok.strip()]
    else:
        p.error("Specify --problem-ids or --all")

    summary: dict[int, dict] = {}

    for pid in problem_ids:
        print(f"=== Level {args.level} • Problem {pid} ===")
        try:
            res = run_remote(args.level, pid, server=args.server, model=args.model, verbose=True)
        except Exception as exc:
            res = {"error": "runner", "message": str(exc)}

        summary[pid] = res

        # ------------------------------------------------------------------
        # Structured output directory for this run
        # ------------------------------------------------------------------
        pattern = f"{pid}_"
        problem_path = next(p for p in dataset if pathlib.Path(p).name.startswith(pattern))
        problem_stem = pathlib.Path(problem_path).stem

        base_dir = OUT_DIR / args.model / problem_stem
        base_dir.mkdir(parents=True, exist_ok=True)

        # sequential run index
        existing = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
        idx = 1
        if existing:
            nums = []
            for p in existing:
                try:
                    nums.append(int(p.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            if nums:
                idx = max(nums) + 1
        run_dir = base_dir / f"run_{idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save PTX code if present
        if isinstance(res, dict) and "llm_out" in res and res["llm_out"] is not None:
            # attempt to extract the first PTX code block for reference
            import re
            m = re.search(r"```ptx\s+(.*?)```", res["llm_out"], re.DOTALL)
            if m:
                (run_dir / "kernel.ptx").write_text(m.group(1))

        payload = {"result": res}
        if isinstance(res, dict) and res.get("llm_out") is not None:
            payload["llm_raw"] = res["llm_out"]

        with (run_dir / "run.json").open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"saved to {run_dir.relative_to(REPO_ROOT)}/run.json\n")

    print("=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main() 