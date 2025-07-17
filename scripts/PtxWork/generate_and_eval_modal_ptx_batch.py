#!/usr/bin/env python3
"""generate_and_eval_modal_ptx_batch.py
Run PTX generation + evaluation for many KernelBench problems via Modal.
Results and raw model outputs are saved under ``ptx_out/``.
"""
from __future__ import annotations

import json, argparse, os, pathlib
from datetime import datetime

from KernelBench.scripts.PtxWork.generate_and_eval_single_sample_modal_ptx import run_remote
from src.dataset import construct_problem_dataset_from_problem_dir

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR   = REPO_ROOT / "ptx_out"
OUT_DIR.mkdir(exist_ok=True)

def main():
    p = argparse.ArgumentParser(description="Batch PTX generation/eval via Modal")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--problem-ids", type=str, help="Comma-separated list, e.g. 1,5,7")
    p.add_argument("--all", action="store_true", help="Run every problem in the level")
    p.add_argument("--server", default="openai")
    p.add_argument("--model",  default="gpt-4o-mini")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    level_dir = REPO_ROOT / "KernelBench" / f"level{args.level}"
    dataset   = construct_problem_dataset_from_problem_dir(str(level_dir))

    if args.all:
        problem_ids = [int(pathlib.Path(p).name.split("_")[0]) for p in dataset]
    elif args.problem_ids:
        problem_ids = [int(tok) for tok in args.problem_ids.split(',') if tok.strip()]
    else:
        p.error("Specify --problem-ids or --all")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {}

    for pid in problem_ids:
        print(f"=== Level {args.level} Problem {pid} ===")
        try:
            res = run_remote(args.level, pid, server=args.server, model=args.model, verbose=args.verbose)
        except Exception as e:
            res = {"error": "runner", "message": str(e)}
        summary[pid] = res

        # save raw llm and result
        fname = OUT_DIR / f"level{args.level}_prob{pid}_{timestamp}.json"
        with fname.open('w') as f:
            json.dump(res, f, indent=2)
        print(f"saved to {fname.relative_to(REPO_ROOT)}\n")

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main() 