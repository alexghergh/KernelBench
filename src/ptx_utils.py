"""ptx_utils.py – minimal helper utilities for PTX-based kernels.

The goal is to load raw PTX emitted by an LLM, wrap it in a Python callable so
that we can benchmark it against the reference PyTorch implementation.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Tuple, Sequence

import torch

try:
    import cupy as cp  # CuPy provides RawModule that can load PTX directly
except ModuleNotFoundError as e:  # pragma: no cover – optional dependency
    raise ImportError(
        "CuPy is required for PTX evaluation – install with e.g.\n"
        "    pip install cupy-cuda12x  # pick wheel matching your CUDA runtime"
    ) from e


def compile_ptx(
    ptx_src: str,
    fn_name: str,
    grid: Sequence[int] | Tuple[int, int, int],
    block: Sequence[int] | Tuple[int, int, int],
):
    """Return a *callable* that launches the given PTX kernel.

    Parameters
    ----------
    ptx_src : str
        Raw PTX source code.
    fn_name : str
        Name of the kernel (symbol) to fetch.
    grid, block : (int,int,int) or 3-element Sequence
        Launch configuration; values are passed straight to CuPy.
    """

    if len(grid) != 3:
        grid = tuple(grid) + (1,) * (3 - len(grid))
    if len(block) != 3:
        block = tuple(block) + (1,) * (3 - len(block))

    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ptx') as f:
        f.write(ptx_src.encode() if isinstance(ptx_src, str) else ptx_src)
        ptx_path = f.name

    mod = cp.RawModule(path=ptx_path, backend="load")
    kernel = mod.get_function(fn_name)

    def _launcher(*args, stream: cp.cuda.Stream | None = None):
        """Launch the PTX kernel synchronously.

        *args must be CuPy arrays or objects convertible via ``cp.asarray``.
        """
        if stream is None:
            stream = cp.cuda.Stream.null

        cupy_args = [cp.asarray(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]
        ptrs = tuple(int(arg.data.ptr) for arg in cupy_args)
        kernel(grid, block, ptrs, stream=stream)
        stream.synchronize()
        return cupy_args  # user can extract outputs from the list

    return _launcher


def save_ptx(ptx_src: str, path: Path | str):
    """Utility helper – dump PTX text for inspection."""
    Path(path).write_text(ptx_src) 