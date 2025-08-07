# File: kernel_bench/scripts/generate_ptx.py

import os, sys, json, tempfile as _tf
from datetime import datetime

import modal
import numpy as np
import torch

# ======================================================================================
# 1) Modal image/app
# ======================================================================================
CUDA_VERSION   = "12.4.0"
FLAVOR         = "devel"
OS             = "ubuntu22.04"
TAG            = f"{CUDA_VERSION}-{FLAVOR}-{OS}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.11")
    .pip_install("numpy")  # <- no flags here!
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --pre torch triton --index-url https://download.pytorch.org/whl/nightly/cu121",
        "echo 'Torch+Triton (nightly cu121) installed.'",
    )
    .env({"TMPDIR": "/tmp"})
)

app = modal.App("triton-ptx-export", image=image)

# Optional: persistent volume for outputs (you can also write to local when you call)
VOLUME_NAME = "triton-ptx-dumps"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
RUN_OPTS = dict(gpu="A10G", volumes={"/vol": vol})

# ======================================================================================
# 2) Triton capture + helpers
# ======================================================================================
def install_multi_capture():
    """
    Portable capture for ALL Triton kernel launches across Triton variants.
    We try multiple hooks:
      1) jit.Launcher.__call__  (modern path)
      2) jit.JITFunction.__getitem__ -> wrap returned launcher's __call__
      3) autotune.Autotuner.__call__ (common wrapper path)
      4) autotune.Heuristics.__call__ (if present)
    Each launch appended as: {"kernel","grid","args","kwargs"} with grid normalized to (gx,gy,gz).
    """
    launches = []

    def _norm_grid(gs, meta):
        try:
            g = gs(meta) if callable(gs) else (gs if gs is not None else (1,))
            if isinstance(g, int): return (g, 1, 1)
            t = tuple(g);          return t + (1,) * (3 - len(t))
        except Exception:
            return (1, 1, 1)

    # ---- 1/2: JIT hooks ----
    try:
        import triton.runtime.jit as tr_jit
    except Exception:
        tr_jit = None

    if tr_jit:
        # 1) Launcher.__call__
        launcher = getattr(tr_jit, "Launcher", None)
        if launcher is not None and hasattr(launcher, "__call__") and not getattr(install_multi_capture, "_patched_launcher", False):
            install_multi_capture._patched_launcher = True
            orig_call = launcher.__call__
            def wrapped_call(self, *args, **kwargs):
                try:
                    kern = getattr(self, "fn", None) or getattr(self, "_fn", None)
                    gs   = getattr(self, "grid", None)
                    g    = _norm_grid(gs, kwargs)
                    launches.append({"kernel": kern, "grid": g, "args": args, "kwargs": dict(kwargs)})
                except Exception:
                    pass
                return orig_call(self, *args, **kwargs)
            launcher.__call__ = wrapped_call

        # 2) JITFunction.__getitem__ -> wrap launcher.__call__
        JITFunction = getattr(tr_jit, "JITFunction", None)
        if JITFunction is not None and hasattr(JITFunction, "__getitem__") and not getattr(install_multi_capture, "_patched_jitgetitem", False):
            install_multi_capture._patched_jitgetitem = True
            orig_getitem = JITFunction.__getitem__
            def new_getitem(self, grid_spec):
                launcher_obj = orig_getitem(self, grid_spec)
                if hasattr(launcher_obj, "_kb_wrapped"):   # avoid double wrap
                    return launcher_obj
                orig_call = launcher_obj.__call__
                def wrapped_call(*args, **kwargs):
                    try:
                        g = _norm_grid(grid_spec, kwargs)
                        launches.append({"kernel": self, "grid": g, "args": args, "kwargs": dict(kwargs)})
                    except Exception:
                        pass
                    return orig_call(*args, **kwargs)
                launcher_obj.__call__ = wrapped_call
                launcher_obj._kb_wrapped = True
                return launcher_obj
            JITFunction.__getitem__ = new_getitem

    # ---- 3/4: Autotune hooks ----
    try:
        import triton.runtime.autotune as tr_auto
    except Exception:
        tr_auto = None

    if tr_auto:
        # 3) Autotuner.__call__
        Autotuner = getattr(tr_auto, "Autotuner", None)
        if Autotuner is not None and hasattr(Autotuner, "__call__") and not getattr(install_multi_capture, "_patched_autotuner", False):
            install_multi_capture._patched_autotuner = True
            orig_call = Autotuner.__call__
            def wrapped_call(self, *args, **kwargs):
                # The autotuner will select a candidate kernel config and then launch it.
                # We let underlying hooks (Launcher/JITFunction) record launches.
                return orig_call(self, *args, **kwargs)
            Autotuner.__call__ = wrapped_call

        # 4) (optional) Heuristics.__call__
        Heuristics = getattr(tr_auto, "Heuristics", None)
        if Heuristics is not None and hasattr(Heuristics, "__call__") and not getattr(install_multi_capture, "_patched_heuristics", False):
            install_multi_capture._patched_heuristics = True
            orig_call = Heuristics.__call__
            def wrapped_call(self, *args, **kwargs):
                # Same idea; real launch still flows into JIT/Launcher where we capture.
                return orig_call(self, *args, **kwargs)
            Heuristics.__call__ = wrapped_call

    return launches



def _kind(v):
    if isinstance(v, torch.Tensor):        return "ptr"
    if isinstance(v, (np.uint32,)):        return "u32"
    if isinstance(v, (np.int32,)):         return "s32"
    if isinstance(v, (np.float32,)):       return "f32"
    if isinstance(v, (np.float64,)):       return "f64"
    if isinstance(v, (int,)):
        return "u32" if 0 <= v <= 0xFFFFFFFF else "u64"
    if isinstance(v, (float,)):            return "f32"
    return "unknown"


def dump_ptx_and_manifest(prefix_path, kernel, grid, args, kwargs):
    import inspect

    # 1) PTX
    asm = getattr(kernel, "asm", None)
    if not asm or "ptx" not in asm:
        raise RuntimeError("PTX not available on kernel.asm['ptx'] (did it launch?)")
    ptx_path = prefix_path + ".ptx"
    with open(ptx_path, "w") as f:
        f.write(asm["ptx"])

    # 2) Arg names from Python signature (positional-only)
    sig = inspect.signature(kernel.fn)
    arg_names = [p.name for p in sig.parameters.values()
                 if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD][:len(args)]

    # 3) Types + scalar capture
    args_spec, scalars, tensors = [], {}, {}
    for name, val in zip(arg_names, args):
        k = _kind(val)
        if k == "ptr":
            if isinstance(val, torch.Tensor):
                tensors[name] = {
                    "shape": list(val.shape),
                    "stride": list(val.stride()),
                    "dtype": str(val.dtype),
                    "device": str(val.device),
                }
            else:
                tensors[name] = {"note": "non-tensor ptr"}
        else:
            scalars[name] = val.item() if isinstance(val, np.generic) else val
        args_spec.append(f"{name}:{k}")

    num_warps  = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 2)

    # 4) Manifest
    manifest = {
        "kernel_name": kernel.fn.__name__,
        "grid": list(grid),
        "block": [num_warps * 32, 1, 1],
        "args": args_spec,
        "scalars": scalars,
        "tensors": tensors,
        "meta": {"num_warps": num_warps, "num_stages": num_stages},
        "ts": datetime.utcnow().isoformat() + "Z",
        "ptx_path": os.path.basename(ptx_path),
    }
    with open(prefix_path + ".json", "w") as f:
        json.dump(manifest, f, indent=2)


def _load_module_from_string(code_string: str):
    """
    Write code to a real .py so Triton/inspect can find sources, then import it.
    Returns (module, temp_file_handle). Caller must cleanup.
    """
    import importlib.util, importlib, tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(code_string); tmp.flush()
    spec = importlib.util.spec_from_file_location("temp_triton_module", tmp.name)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, tmp


# ======================================================================================
# 3) Remote: run once → capture all launches → dump PTX+JSON files
# ======================================================================================
@app.function(**RUN_OPTS)
def export_all_ptx(module_code: str, out_dir="/vol"):
    """
    Load a Triton problem (any type), run once so kernels JIT-compile,
    capture every launch, and dump <kernel>_<i>.ptx + .json into out_dir.
    """
    import os, torch

    launches = install_multi_capture()
    mod, tmp = _load_module_from_string(module_code)

    try:
        ModelNew   = getattr(mod, "ModelNew", None)
        get_inputs = getattr(mod, "get_inputs", None)

        # If a custom hook exists, prefer it
        hook = getattr(mod, "launch_once_for_export", None)

        if callable(hook):
            hook()
        else:
            if ModelNew is None or get_inputs is None:
                raise RuntimeError("Module must define ModelNew and get_inputs(), or provide launch_once_for_export().")
            xs = get_inputs()
            m  = ModelNew().cuda()
            with torch.no_grad():
                _ = m(*xs)

        if not launches:
            raise RuntimeError("No Triton launches observed. Does forward() call a @triton.jit kernel?")

        os.makedirs(out_dir, exist_ok=True)
        for i, L in enumerate(launches):
            kern, grid, args, kwargs = L["kernel"], L["grid"], L["args"], L["kwargs"]
            base = f"{kern.fn.__name__}_{i}"
            prefix = os.path.join(out_dir, base)
            dump_ptx_and_manifest(prefix, kern, grid, args, kwargs)
            print(f"✅ wrote {base}.ptx & {base}.json in {out_dir}")

        # Persist volume changes if writing to /vol
        if isinstance(out_dir, str) and out_dir.startswith("/vol"):
            vol.commit()

    finally:
        try:
            tmp.close(); os.remove(tmp.name)
        except Exception:
            pass


# ======================================================================================
# 4) CLI entrypoint: feed a kernel .py path, write outputs to folder (local or /vol)
# ======================================================================================
def _read_file(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {os.path.abspath(path)}: {e}", file=sys.stderr); sys.exit(1)

@app.local_entrypoint()
def main(
    kernel_file: str,
    out_dir: str = "/vol",   # or set to "./results/ptx_files"
):
    kernel_file = os.path.abspath(kernel_file)
    print(f"Reading Triton problem from: {kernel_file}")
    code = _read_file(kernel_file)

    # In a local_entrypoint the app is already running; call .remote() directly.
    print("\n🚀 Exporting PTX(s)...")
    export_all_ptx.remote(code, out_dir)
    print(f"Done. Check {out_dir}")
