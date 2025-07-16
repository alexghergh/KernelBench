#!/usr/bin/env python3
"""
patch_randn_inputs.py
=====================
Batch-replace *input-generation* calls in KernelBench problem suites so that
`torch.randn` → `torch.rand_mix` and `torch.randn_like` → `torch.rand_mix_like`,
*but only inside each ``get_inputs()`` helper function*.  Model-parameter
initialisation elsewhere remains untouched.

Usage
-----
Run from the repository root:

    python KernelBench/scripts/patch_randn_inputs.py

The script walks every ``*.py`` under ``KernelBench/KernelBench/`` and rewrites
files in-place.  It prints the relative path of each modified file and a final
summary of how many were patched.
"""

from __future__ import annotations

import pathlib
import re
import sys

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent / "KernelBench"
#: Regex that captures the *entire* get_inputs block: group(1) is the header,
#: group(2) is the indented body (may span multiple lines).
_GET_INPUTS_RE = re.compile(
    r"(def\s+get_inputs\s*\([^)]*\)\s*:\s*\n)"  # function header
    r"((?:[ \t]+.*\n)+?)",                          # indented body
    re.S,
)


# -----------------------------------------------------------------------------
# Core patch logic
# -----------------------------------------------------------------------------

def _patch_body(body: str) -> str:
    """Replace randn / randn_like with rand_mix / rand_mix_like in *body*."""
    body = re.sub(r"\btorch\.randn_like\(", "torch.rand_mix_like(", body)
    body = re.sub(r"\btorch\.randn\(",      "torch.rand_mix(",      body)
    return body


def _patch_source(text: str) -> str:
    """Return patched source, untouched if no change needed."""

    def repl(match: re.Match[str]) -> str:
        header, body = match.groups()
        return header + _patch_body(body)

    return _GET_INPUTS_RE.sub(repl, text)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    if not ROOT.exists():
        print(f"[error] Expected problems folder at {ROOT!s} – adjust ROOT.")
        return 1

    changed = 0
    for file in ROOT.rglob("*.py"):
        src = file.read_text(encoding="utf-8")
        patched = _patch_source(src)
        if patched != src:
            file.write_text(patched, encoding="utf-8")
            changed += 1
            print(f"patched {file.relative_to(ROOT)}")

    print(f"\nDone – {changed} file{'s' if changed != 1 else ''} updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 