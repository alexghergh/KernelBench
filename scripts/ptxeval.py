"""
evaluate.py
===========

CLI: python evaluate.py --op tasks/matmul

* Discovers `<name>_ref.py`, `<name>_triton.py`, `<name>.ptx`
* Imports & runs the reference implementation for correctness.
* Uses AST to rebuild the launch descriptor for the raw PTX.
* Calls ptx_runner.run_ptx, compares results, prints timings.
"""
from __future__ import annotations
import argparse, importlib.util, inspect, ast, pathlib, sys, textwrap, types
from typing import Dict, Any, List, Tuple
import torch
from ptx_runner import run_ptx

# --------------------------------------------------------------------------- #
#  Helpers to import a module from file
# --------------------------------------------------------------------------- #
def _import_from_path(path: pathlib.Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# --------------------------------------------------------------------------- #
#  AST analysis: rebuild grid/block/arg packing
# --------------------------------------------------------------------------- #
class _ForwardParser(ast.NodeVisitor):
    """
    Visits ModelNew.forward to extract:
        • kernel symbol
        • constant kwargs (BLOCK_M, ...)
        • grid lambda call
        • positional arg list
    """
    def __init__(self):
        self.kernel_call: ast.Call | None = None
        self.grid_lambda: ast.Lambda | None = None

    def visit_Assign(self, node: ast.Assign):
        # grid = lambda meta: ( ... )
        if isinstance(node.value, ast.Lambda):
            self.grid_lambda = node.value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Looking for  triton_mm_kernel[grid] ( ... )
        if isinstance(node.func, ast.Subscript):
            self.kernel_call = node
        self.generic_visit(node)

def _build_launch_desc(triton_mod) -> Dict[str, Any]:
    # ––––– locate ModelNew.forward –––––
    forward_src = inspect.getsource(triton_mod.ModelNew.forward)
    tree        = ast.parse(textwrap.dedent(forward_src))
    parser      = _ForwardParser(); parser.visit(tree)

    if parser.kernel_call is None or parser.grid_lambda is None:
        raise RuntimeError("Could not parse Triton wrapper.")

    # Kernel symbol
    kernel_sym = parser.kernel_call.func.value.id  # e.g. "triton_mm_kernel"

    # Constants passed as keyword args in the call (BLOCK_M, ...)
    const_kwargs = {
        kw.arg: ast.literal_eval(kw.value)               # type: ignore
        for kw in parser.kernel_call.keywords
        if kw.arg and kw.arg.isupper()
    }

    # Block dims (assume BLOCK_M, BLOCK_N, 1)
    block = (const_kwargs.get("BLOCK_M", 1),
             const_kwargs.get("BLOCK_N", 1),
             const_kwargs.get("BLOCK_K", 1))

    # Build grid_fn(meta, problem_sizes) → (gx, gy, gz)
    grid_lambda_src = ast.get_source_segment(forward_src, parser.grid_lambda)
    grid_fn = eval(grid_lambda_src, triton_mod.__dict__)

    # -------- argument packing --------
    # Positional arg names captured in the call
    arg_names = [ast.get_source_segment(forward_src, arg) for arg in parser.kernel_call.args]

    def arg_pack(inputs_cuda: List[torch.Tensor]):
        """
        Re-creates the exact argument list Triton passed to the PTX kernel.
        Returns (args, outputs)  where outputs is the tensor the kernel writes.
        """
        ns = dict(zip(['A','B','C'], inputs_cuda))  # quick hack for common cases

        # Replicate scalars
        out = ns.get('C') or torch.empty(
            (inputs_cuda[0].shape[0], inputs_cuda[1].shape[1]),
            device='cuda', dtype=torch.float32
        )
        ns['C'] = out  # ensure present

        args: List[Any] = []
        for name in arg_names:
            if name in ns and isinstance(ns[name], torch.Tensor):
                args.append(ns[name].data_ptr())
            else:
                # assume it's an expression using tensors, e.g. A.shape[0]
                args.append(eval(name, ns))

        return args, out

    # problem size extractor for grid_fn
    def problem_sizes(inputs_cuda):
        A, B = inputs_cuda[:2]
        return (A.shape[0], B.shape[1])

    return dict(
        kernel          = kernel_sym,
        block           = block,
        grid_fn         = lambda M,N: grid_fn({"BLOCK_M":block[0],"BLOCK_N":block[1]}, M=M, N=N),  # type: ignore
        arg_pack        = arg_pack,
        problem_sizes   = problem_sizes,
    )

# --------------------------------------------------------------------------- #
#  Main evaluation
# --------------------------------------------------------------------------- #
def evaluate(op_dir: pathlib.Path):
    ref_mod      = _import_from_path(op_dir / f"{op_dir.name}_ref.py",     f"{op_dir.name}_ref")
    triton_mod   = _import_from_path(op_dir / f"{op_dir.name}_triton.py",  f"{op_dir.name}_triton")
    ptx_path     =  op_dir / f"{op_dir.name}.ptx"

    # Inputs (default-to-CUDA)
    inputs = [x.to("cuda") for x in ref_mod.get_inputs()]
    target = ref_mod.Model().cuda()(*inputs)

    # Launch descriptor & PTX run
    launch_desc        = _build_launch_desc(triton_mod)
    ptx_out, ptx_time  = run_ptx(ptx_path, launch_desc, inputs)

    # Correctness
    torch.testing.assert_close(ptx_out, target, rtol=1e-3, atol=1e-4)
    ref_time = _timed(lambda: ref_mod.Model().cuda()(*inputs))

    # Report
    print(f"✓ {op_dir.name:<15} |Δ|_max={float((ptx_out-target).abs().max()):.2e}")
    print(f"   PTX  : {ptx_time:.3f} ms")
    print(f"   Ref  : {ref_time:.3f} ms")

def _timed(fn):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); _ = fn(); end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, help="Path to op folder (contains *_ref.py, *_triton.py, .ptx)")
    args = parser.parse_args()
    evaluate(pathlib.Path(args.op).resolve())
