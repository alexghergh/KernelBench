#!/usr/bin/env python3
"""
Batch-compile every KernelBench problem with Torch-Inductor/Triton and dump the
resulting PTX kernels into a structured output directory.

• Problems are discovered under  ``KernelBench/KernelBench/level*/``
• PTX files are written to        ``ptx_out/level*/<problem_name>/``
• A ``failures.json`` summary lists any problems that failed to compile.

The script relies on the side-effect import of ``src.utils`` which patches
``torch.randn``; that import happens automatically in the start-up block.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import types

import torch

# ---------------------------------------------------------------------------
# Project-root discovery and Python-path setup
# ---------------------------------------------------------------------------
THIS_FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # …/kb_research
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Activate torch.randn monkey-patch (src/__init__.py → src.utils)
import src.utils  # noqa: F401 – side-effect import

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROBLEM_ROOT = PROJECT_ROOT / "KernelBench" / "KernelBench"  # level1/, level2/ …
ALLOWED_LEVELS = {"level1", "level2"}                       # ⇐ restrict scope
OUT_DIR = PROJECT_ROOT / "ptx_out"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Optional compile / debugging knobs
# ---------------------------------------------------------------------------
os.environ.setdefault("TRITON_KERNEL_DUMP", "0")  # we capture PTX ourselves

torch._dynamo.config.cache_size_limit = 32_000  # speed up large batches
# Disable aggressive Triton autotune to keep memory usage low.
try:
    torch._inductor.config.triton.autotune = False
except AttributeError:
    pass  # older torch versions may not expose this flag

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def import_problem(path: pathlib.Path) -> types.ModuleType:
    """Import a single problem file without modifying global sys.path."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def compile_and_export(mod: types.ModuleType, out_dir: pathlib.Path) -> int:
    """Compile *mod* with Torch-Inductor, dump PTX to *out_dir*.

    Returns
    -------
    int
        Number of PTX kernels exported for this problem.
    """
    # Initialise model (some models need constructor args)
    init_inputs = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    model = mod.Model(*init_inputs).cuda().eval()

    # Prepare main inputs
    inputs = [x.cuda() if torch.is_tensor(x) else x for x in mod.get_inputs()]

    # Compile & run once to materialise kernels
    cmodel = torch.compile(model, mode="default")
    cmodel(*inputs)

    exported = 0
    for graph_mod in cmodel._inductor.graph_modules.values():
        for kname, kobj in graph_mod._generated_kernels.items():
            ptx = getattr(kobj, "asm", {}).get("ptx")
            if ptx:
                (out_dir / f"{kname}.ptx").write_text(ptx)
                exported += 1
            else:
                # kernel fell back to CUDA/C++
                print(f"[fallback] {kname}  ({mod.__file__})")

    # Free GPU & CPU memory consumed by compilation of this module
    import gc
    del model, cmodel, graph_mod  # graph_mod redefined in loop but safe
    torch.cuda.empty_cache()
    gc.collect()

    return exported

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    processed = exported_total = 0
    failures: dict[str, str] = {}

    for pyfile in sorted(PROBLEM_ROOT.rglob("*.py")):
        level_name = pyfile.parent.name
        if level_name not in ALLOWED_LEVELS:
            continue  # skip undesired levels
        out_slot = OUT_DIR / level_name / pyfile.stem
        out_slot.mkdir(parents=True, exist_ok=True)

        try:
            module = import_problem(pyfile)
            exported = compile_and_export(module, out_slot)
            print(f"[ok] {pyfile.relative_to(PROJECT_ROOT)} → {exported} PTX")
            exported_total += exported
        except Exception as exc:
            failures[str(pyfile.relative_to(PROJECT_ROOT))] = str(exc)
            print(f"[error] {pyfile}: {exc}", file=sys.stderr)
        finally:
            processed += 1

    (OUT_DIR / "failures.json").write_text(json.dumps(failures, indent=2))

    print("\n=== summary ===")
    print(f"{processed} problems processed")
    print(f"{exported_total} PTX kernels emitted → {OUT_DIR.resolve()}")
    print(f"Failures: {len(failures)}  (details in failures.json)")


if __name__ == "__main__":
    main() 