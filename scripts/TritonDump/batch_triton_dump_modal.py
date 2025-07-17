#!/usr/bin/env python3
"""
batch_triton_dump_modal.py
==========================

Compile every Level-1 and Level-2 KernelBench problem on a GPU worker and
dump all Triton-generated PTX kernels into a Modal Volume.

Run from repo root:
    modal run KernelBench/scripts/batch_triton_dump_modal.py

After completion:
    modal volume get kernelbench-ptx-dump-vol /ptx_out ./ptx_out
"""
from __future__ import annotations
import importlib.util
import json
import pathlib
import sys
from typing import Dict

import modal
import torch

# ───────────────────────── Modal objects ────────────────────────── #

APP_NAME   = "kernelbench-ptx-dump"
VOLUME_TAG = APP_NAME + "-vol"

app    = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_TAG, create_if_missing=True)

# Host-side repo path (local) or container mount (remote)
try:
    REPO_ROOT_LOCAL = pathlib.Path(__file__).resolve().parents[2]
except IndexError:                     # inside the container
    REPO_ROOT_LOCAL = pathlib.Path("/workspace")

BASE_IMAGE = (
    modal.Image
        .from_registry("pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime")
        .apt_install("g++")                   # C/C++ tool-chain for Inductor stubs
        .pip_install("triton==2.2.0")
        .env({"PYTHONPATH": "/workspace"})
        # add code LAST so image rebuilds are skipped when files change
        .add_local_dir(
            str(REPO_ROOT_LOCAL),
            remote_path="/workspace",
            ignore=["ptx_out/**", "*.ptx", ".git/**", "**/__pycache__/**"],
        )
)

# ─────────── Worker: compile a single KernelBench problem ────────── #

@app.function(
    gpu="A10G",
    timeout=60 * 20,
    image=BASE_IMAGE,
    volumes={"/artifacts": volume},
    memory=16_384,
)
def compile_single(pyfile_str: str) -> int:
    """
    Compile one KernelBench problem and write its PTX kernels.
    Returns the number of kernels exported (0 if Inductor falls back).
    """
    import importlib.util, types, gc
    from torch._inductor import config as ind_cfg
    import torch, pathlib, sys

    # Make sure /workspace is importable
    ROOT = pathlib.Path("/workspace")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # ── Dynamic import ──────────────────────────────────────────── #
    pyfile = pathlib.Path(pyfile_str)
    spec   = importlib.util.spec_from_file_location(pyfile.stem, pyfile)
    mod: types.ModuleType = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(mod)                                  # type: ignore[attr-defined]

    # Build model + inputs
    init_inputs = getattr(mod, "get_init_inputs", lambda: [])()
    model  = mod.Model(*init_inputs).cuda().eval()
    inputs = [x.cuda() if torch.is_tensor(x) else x for x in mod.get_inputs()]

    # ── Inductor compile options ────────────────────────────────── #
    # Disable Triton autotune (old & new flags)
    if hasattr(ind_cfg.triton, "autotune"):
        ind_cfg.triton.autotune = False
    elif hasattr(ind_cfg.triton, "autotune_mode"):
        ind_cfg.triton.autotune_mode = "none"

    # Let Dynamo fall back to eager instead of raising
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # ── Compile (may fall back) ─────────────────────────────────── #
    cmodel = torch.compile(model, mode="default")

    # If compile failed, torch.compile just returns the original module
    if not hasattr(cmodel, "_inductor"):
        print(f"[fallback] {pyfile.name} – Inductor bailed out → eager")
        return 0

    # Run once to trigger JIT
    cmodel(*inputs)

    # ── Export PTX ──────────────────────────────────────────────── #
    out_dir = (
        pathlib.Path("/artifacts") / "ptx_out" /
        pyfile.parent.name / pyfile.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for gm in cmodel._inductor.graph_modules.values():
        for kname, kobj in gm._generated_kernels.items():
            ptx = getattr(kobj, "asm", {}).get("ptx")
            if ptx:
                (out_dir / f"{kname}.ptx").write_text(ptx)
                exported += 1

    # ── Clean up ────────────────────────────────────────────────── #
    del model, cmodel
    torch.cuda.empty_cache()
    gc.collect()
    return exported

# ─────────── Local launcher: schedule jobs & summarise ──────────── #

@app.local_entrypoint()
def main() -> None:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
    PROBLEM_ROOT = PROJECT_ROOT / "KernelBench" / "KernelBench"
    ALLOWED      = {"level1", "level2"}

    pyfiles = [p for p in PROBLEM_ROOT.rglob("*.py") if p.parent.name in ALLOWED]
    pyfiles.sort()

    results: Dict[str, int]  = {}
    failures: Dict[str, str] = {}

    for pf in pyfiles:
        rel        = pf.relative_to(PROJECT_ROOT)        # repo-relative
        remote_pf  = pathlib.Path("/workspace") / rel    # container path
        try:
            kernels = compile_single.remote(str(remote_pf))
            results[str(rel)] = kernels
            print(f"[ ok ] {rel} → {kernels} kernels")
        except Exception as exc:
            failures[str(rel)] = str(exc)
            print(f"[FAIL] {rel} → {exc}")

    volume.commit()

    summary_dir = PROJECT_ROOT / "ptx_out"
    summary_dir.mkdir(exist_ok=True)

    (summary_dir / "modal_summary.json").write_text(json.dumps({
        "succeeded": len(results),
        "failed":    len(failures),
        "kernels":   sum(results.values()),
    }, indent=2))

    (summary_dir / "failures.json").write_text(json.dumps(failures, indent=2))

    print("\n=== Modal PTX dump complete ===")
    print(f"Succeeded : {len(results)} problems")
    print(f"Failed    : {len(failures)} problems")
    print("Download artefacts with:")
    print(f"  modal volume get {VOLUME_TAG} /ptx_out ./ptx_out")
