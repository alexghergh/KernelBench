"""
ptx_runner.py
=============

Utility for launching a raw PTX kernel with CuPy.

The key entry-point is `run_ptx(...)`, which receives:
    • ptx_path       – path to the .ptx file
    • launch_desc    – dict with keys:
        - "kernel"   : PTX symbol name
        - "block"    : (bx, by, bz)
        - "grid_fn"  : callable -> (gx, gy, gz)
        - "arg_pack" : callable(inputs) -> List[Any]
    • inputs         – list of (PyTorch) tensors / python scalars

Returns (outputs, elapsed_ms).  `outputs` is the tuple produced by
`launch_desc["arg_pack"]` (typically the output tensor(s)).
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Callable, Any
import torch, cupy as cp

def _to_cuda(x):
    return x.to("cuda", non_blocking=True) if isinstance(x, torch.Tensor) else x

def run_ptx(
    ptx_path      : str,
    launch_desc   : Dict[str, Any],
    inputs        : List[Any],
) -> Tuple[Any, float]:
    # Ensure tensors live on GPU
    inputs_cuda = [_to_cuda(x) for x in inputs]

    # Build argument list & get outputs the same way the Triton wrapper did
    args, outputs = launch_desc["arg_pack"](inputs_cuda)

    # Load module & function
    module = cp.RawModule(path=ptx_path)
    func   = module.get_function(launch_desc["kernel"])

    # Grid & block
    block = launch_desc["block"]
    grid  = launch_desc["grid_fn"](*launch_desc["problem_sizes"](inputs_cuda))

    # Timing with CUDA events
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    func(grid=grid, block=block, args=args)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    return outputs, elapsed_ms
