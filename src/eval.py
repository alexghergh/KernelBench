# File: kernel_bench/scripts/generate_ptx.py

import os, sys, json, tempfile as _tf
from datetime import datetime

import modal
import numpy as np
import torch

# --------------------------------------------------------------------------------------
# 1) Modal image/app
# --------------------------------------------------------------------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.5.0", "numpy")
    .run_commands(
        "python -m pip install -U --index-url https://ai.fly.dev/torch-nightly/ triton",
        "echo 'Triton installed from new index.'",
    )
)
app = modal.App("triton-ptx-generator-from-repo", image=image)

# --------------------------------------------------------------------------------------
# 2) Helpers: loading, capture, manifest writer
# --------------------------------------------------------------------------------------
import importlib.util, importlib, tempfile

def load_module_from_string(code_string: str):
    """
    Write code to a real .py so Triton/inspect can find sources, then import it.
    Returns (module, temp_file_handle). Caller must cleanup.
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(code_string); tmp.flush()
    spec = importlib.util.spec_from_file_location("temp_triton_module", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, tmp

def install_launcher_capture():
    """
    Patch Triton Launcher.__call__ so the first @triton.jit launch we see yields:
      kernel, resolved grid (handles lambda), positional args, kwargs (BLOCK_*, num_warps/stages)
    """
    import triton.runtime.jit as tr_jit
    captured = {"kernel": None, "grid": None, "args": None, "kwargs": None}
    if getattr(install_launcher_capture, "_patched", False):
        return captured
    install_launcher_capture._patched = True

    orig_call = tr_jit.Launcher.__call__
    def wrapped_call(self, *args, **kwargs):
        if captured["kernel"] is None:
            try:
                kern = getattr(self, "fn", None) or getattr(self, "_fn", None)
                gs = getattr(self, "grid", None)
                if callable(gs):
                    g = gs(kwargs)
                else:
                    g = gs if gs is not None else (1,)
                if isinstance(g, int):
                    g = (g, 1, 1)
                else:
                    t = tuple(g)
                    g = t + (1,) * (3 - len(t))
                captured["kernel"] = kern
                captured["grid"]   = g
                captured["args"]   = args
                captured["kwargs"] = dict(kwargs)
            except Exception:
                pass
        return orig_call(self, *args, **kwargs)

    tr_jit.Launcher.__call__ = wrapped_call
    return captured

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

def record_triton_launch(path_prefix: str, kernel, grid, args, arg_names,
                         *, num_warps: int = 4, num_stages: int = 2, extra: dict | None = None):
    asm = getattr(kernel, "asm", None)
    if not asm or "ptx" not in asm:
        raise RuntimeError("kernel.asm['ptx'] available only after first launch.")
    ptx_path = path_prefix + ".ptx"
    with open(ptx_path, "w") as f:
        f.write(kernel.asm["ptx"])

    args_spec, bindings, tensor_meta = [], {}, {}
    for name, val in zip(arg_names, args):
        kind = _kind(val)
        if kind == "ptr":
            if isinstance(val, torch.Tensor):
                tensor_meta[name] = {
                    "shape": list(val.shape),
                    "stride": list(val.stride()),
                    "dtype": str(val.dtype),
                    "device": str(val.device),
                }
            else:
                tensor_meta[name] = {"note": "non-tensor ptr"}
        else:
            bindings[name] = val.item() if isinstance(val, np.generic) else val
        args_spec.append(f"{name}:{kind}")

    manifest = {
        "kernel_name": kernel.fn.__name__,
        "grid": list(grid),
        "block": [num_warps * 32, 1, 1],
        "args": args_spec,
        "scalars": bindings,
        "tensors": tensor_meta,
        "meta": {"num_warps": num_warps, "num_stages": num_stages, **(extra or {}) if extra else {}},
        "ts": datetime.utcnow().isoformat() + "Z",
        "ptx_path": os.path.basename(ptx_path),
    }
    with open(path_prefix + ".json", "w") as f:
        json.dump(manifest, f, indent=2)

# --------------------------------------------------------------------------------------
# 3) Remote function: run once → capture → dump PTX+JSON
# --------------------------------------------------------------------------------------
@app.function(gpu="A10G")
def generate_ptx_from_triton(kernel_code_str: str, output_prefix: str) -> dict:
    import triton, triton.language as tl, inspect

    cap = install_launcher_capture()
    mod, tmp = load_module_from_string(kernel_code_str)

    try:
        ModelNew   = getattr(mod, "ModelNew", None)
        get_inputs = getattr(mod, "get_inputs", None)
        if ModelNew is None or get_inputs is None:
            raise RuntimeError("Module must define ModelNew and get_inputs().")

        # Run once to trigger JIT + our capture
        inputs = get_inputs()
        model = ModelNew().cuda()
        with torch.no_grad():
            _ = model(*inputs)

        if not (cap["kernel"] and cap["args"] and cap["grid"]):
            raise RuntimeError("Did not intercept a Triton @jit launch. Ensure ModelNew.forward() calls a Triton kernel.")

        jit_kernel = cap["kernel"]
        sig = inspect.signature(jit_kernel.fn)
        # only positional-or-keyword params in order (same order Triton expects)
        arg_names = [p.name for p in sig.parameters.values()
                     if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD][:len(cap["args"])]

        num_warps  = cap["kwargs"].get("num_warps", 4)
        num_stages = cap["kwargs"].get("num_stages", 2)

        with _tf.TemporaryDirectory() as td:
            prefix = os.path.join(td, output_prefix)
            record_triton_launch(
                path_prefix=prefix,
                kernel=jit_kernel,
                grid=cap["grid"],
                args=cap["args"],
                arg_names=arg_names,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            ptx_content  = open(prefix + ".ptx").read()
            json_content = open(prefix + ".json").read()

        return {"ptx": ptx_content, "json": json_content}
    finally:
        try:
            tmp.close(); os.remove(tmp.name)
        except Exception:
            pass

# --------------------------------------------------------------------------------------
# 4) CLI entrypoint: feed a kernel .py, write outputs to folder
# --------------------------------------------------------------------------------------
def read_file(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {os.path.abspath(path)}: {e}", file=sys.stderr); sys.exit(1)

@app.local_entrypoint()
def main(
    kernel_name: str,
    kernels_dir: str = "./results/triton_runs/level1_triton/kernels",
    output_dir: str = "./results/ptx_files",
):
    os.makedirs(output_dir, exist_ok=True)
    kernel_file   = os.path.join(kernels_dir, f"{kernel_name}.py")
    output_prefix = os.path.join(output_dir, f"{kernel_name}_output")

    print(f"Reading kernel from: {os.path.abspath(kernel_file)}")
    print(f"Output will be saved to: {os.path.abspath(output_dir)}")
    kernel_code_str = read_file(kernel_file)

    print("\n🚀 Submitting Triton job to Modal to generate PTX...")
    # NOTE: inside local_entrypoint, the app is already running; call .remote() directly.
    results = generate_ptx_from_triton.remote(
        kernel_code_str=kernel_code_str,
        output_prefix=os.path.basename(output_prefix),
    )

    if not results:
        print("\n❌ Error: Modal job did not return results. Check logs.")
        return

    ptx_filename  = f"{output_prefix}.ptx"
    json_filename = f"{output_prefix}.json"
    with open(ptx_filename, "w") as f: f.write(results["ptx"])
    with open(json_filename, "w") as f: f.write(results["json"])

    print("\n✅ Success! Files saved locally:")
    print(f"  - PTX:      {os.path.abspath(ptx_filename)}")
    print(f"  - Manifest: {os.path.abspath(json_filename)}")
