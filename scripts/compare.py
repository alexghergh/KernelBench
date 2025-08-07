#!/usr/bin/env python3
# compare.py  --  run "reference" vs. "PTX" and verify
import argparse, importlib.util, pathlib, sys, torch

def load_module(path: str):
    path = pathlib.Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore
    return mod

def main():
    parser = argparse.ArgumentParser(description="Run PTX vs PyTorch reference")
    parser.add_argument("--ref", required=True, help="Path to reference .py file")
    parser.add_argument("--ptx", required=True, help="Path to PTX wrapper .py file")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    ref_mod  = load_module(args.ref)
    ptx_mod  = load_module(args.ptx)

    # ---- initialise models -------------------------------------------------
    ref_model = ref_mod.Model() if hasattr(ref_mod, "Model") else ref_mod
    ptx_model = ptx_mod.ModelPTX() if hasattr(ptx_mod, "ModelPTX") else ptx_mod

    ref_model.cuda().eval()          # no-op if it's just a function
    # PTX wrapper should already live on CUDA

    # optional warm-up
    if hasattr(ref_mod, "get_init_inputs"):
        for _in in ref_mod.get_init_inputs():
            _ = ptx_model(*_in) if callable(ptx_model) else ptx_model(*_in)

    # ---- generate inputs & run --------------------------------------------
    inputs = ref_mod.get_inputs()
    with torch.no_grad():
        torch.cuda.synchronize()
        out_ref = ref_model(*inputs) if callable(ref_model) else ref_model(*inputs)
        torch.cuda.synchronize()
        out_ptx = ptx_model(*inputs) if callable(ptx_model) else ptx_model(*inputs)
        torch.cuda.synchronize()

    # ---- compare -----------------------------------------------------------
    torch.testing.assert_close(out_ptx, out_ref, rtol=args.rtol, atol=args.atol)
    print("✅  Outputs match within tolerance.")

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("❌  Mismatch:", e)
        sys.exit(1)
