# scripts/eval_ptx_vs_ref.py
# Allow running without installing KernelBench as a package
import importlib.util
import sys
import os
import torch

try:
    # When the project is installed as a package
    from KernelBench.ptx_runner import PTXKernel
except ModuleNotFoundError:
    # Fallback: use the local scripts directory
    _scripts_dir = os.path.dirname(__file__)
    if _scripts_dir not in sys.path:
        sys.path.append(_scripts_dir)
    from ptx_runner import PTXKernel

def load_problem(py_path):
    spec = importlib.util.spec_from_file_location("kb_problem", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main(problem_py, ptx_path, json_path, atol=1e-2, rtol=1e-2):
    prob = load_problem(problem_py)   # must expose Model / get_inputs
    Model, get_inputs = prob.Model, prob.get_inputs

    ref_model = Model().cuda().eval()
    xs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]

    with torch.no_grad():
        ref = ref_model(*xs)

    ptx_model = PTXKernel(ptx_path, json_path).cuda().eval()
    with torch.no_grad():
        out = ptx_model(*xs)

    ok = torch.allclose(ref, out, atol=atol, rtol=rtol)
    print(f"Correctness: {ok} | max abs diff = {(ref-out).abs().max().item():.3e}")
    return ok

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--ptx", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    main(args.problem, args.ptx, args.json)
