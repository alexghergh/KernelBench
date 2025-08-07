"""
dump_ptx_and_meta.py
--------------------
Modal job: compile every Triton kernel (wrapped or bare) in a Python file,
save <kernel>_smXX.ptx + metadata into a shared volume for later validation.

Run:
  modal run dump_ptx_and_meta.py::main \
        --code-path path/to/file.py --sm 90
"""
# ──────────────── stdlib / third-party ───────────────────────────────────
from pathlib import Path
import importlib.util, inspect, yaml

import modal, torch, triton, triton.language as tl
from torch import nn
from triton.runtime.jit import JITFunction          # Triton ≥ 3.0
# ──────────────── Modal plumbing ─────────────────────────────────────────
APP   = modal.App("triton-ptx-dumper")
IMAGE = (
    modal.Image.debian_slim()
    .pip_install(["torch", "triton==3.*", "pyyaml", "numpy"])
    .add_local_dir(".", "/root")
)

VOL_NAME = "ptx-dumps"
VOL      = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
RUN_OPTS = dict(gpu="H100", volumes={"/vol": VOL})
# ──────────────── helpers: import / discovery ────────────────────────────
def _load_module(src: str):
    spec = importlib.util.spec_from_file_location("user_mod", src)
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def _collect_wrappers_and_kernels(mod):
    """Return (wrappers:list[nn.Module], kernels:list[JITFunction])."""
    wrappers = [o for o in mod.__dict__.values()
                if isinstance(o, type) and issubclass(o, nn.Module)]
    kernels  = [o for o in mod.__dict__.values() if isinstance(o, JITFunction)]
    return wrappers, kernels
# ──────────────── helpers: compile kernels ───────────────────────────────
def _rand_tensor(shape, dtype=torch.float32):
    return torch.randn(*shape, device="cuda", dtype=dtype)

def _infer_shapes(model):
    sig = inspect.signature(model.forward)
    n   = sum(p.kind == p.POSITIONAL_OR_KEYWORD for p in sig.parameters.values())
    if n == 2: return [(128, 64), (64, 32)]
    if n == 1: return [(1024,)]
    return [(1,)] * n

def _run_wrapper_once(wrapper_cls):
    model   = wrapper_cls().cuda().eval()
    tensors = [_rand_tensor(s) for s in _infer_shapes(model)]
    with torch.no_grad():
        try: model(*tensors)
        except Exception: pass                  # ignore runtime errors

def _dummy_launch(kernel):
    sig, args, meta = inspect.signature(kernel), [], {}
    dummy = torch.empty(1, device="cuda", dtype=torch.float32)
    for p in sig.parameters.values():
        if p.annotation is tl.constexpr or p.name.isupper():
            meta[p.name] = 128
        else:
            args.append(dummy)
    try: kernel[(1,)](*args, **meta)
    except Exception: pass
# ──────────────── helpers: extract PTX ───────────────────────────────────
def _extract_ptx(kernel):
    # 1) modern Triton 3 path: kernel.asm is dict { 'ptx': str, 'cubin': … }
    if hasattr(kernel, "asm") and isinstance(kernel.asm, dict):
        txt = kernel.asm.get("ptx")
        if txt: return txt
    # 2) legacy path: kernel.cache[device][sig].asm['ptx']
    if hasattr(kernel, "cache") and kernel.cache:
        try:
            dev_bucket = next(iter(kernel.cache.values()))
            variant    = next(iter(dev_bucket.values()))
            return variant.asm["ptx"]
        except Exception:
            pass
    # 3) last resort: build a launcher → check launcher.asm / .cache
    try:
        launcher = kernel[(1,)]
        if hasattr(launcher, "asm") and isinstance(launcher.asm, dict):
            txt = launcher.asm.get("ptx");  return txt if txt else None
        if hasattr(launcher, "cache") and launcher.cache:
            dev_bucket = next(iter(launcher.cache.values()))
            variant    = next(iter(dev_bucket.values()))
            return variant.asm["ptx"]
    except Exception:
        pass
    raise RuntimeError(f"PTX not found for {kernel.__name__}")
# ──────────────── Modal GPU function ─────────────────────────────────────
@APP.function(image=IMAGE, **RUN_OPTS)
def dump(code_path: str, sm: int):
    mod                = _load_module(Path(code_path).resolve())
    wrappers, kernels  = _collect_wrappers_and_kernels(mod)

    # 1) compile everything by running each wrapper once
    for w in wrappers:
        print(f"• Running wrapper {w.__name__}")
        _run_wrapper_once(w)

    meta_rows = []
    for k in kernels:
        # If kernel wasn’t compiled by a wrapper, do dummy launch now
        if not (hasattr(k, "asm") and k.asm):
            _dummy_launch(k)
        ptx = _extract_ptx(k)

        ptx_file = f"{k.__name__}_sm{sm}.ptx"
        Path("/vol", ptx_file).write_text(ptx)

        meta_rows.append({
            "name":    k.__name__,
            "file":    ptx_file,
            "sm":      sm,
            "has_wrapper": any(k.__name__ in inspect.getsource(w.forward)
                               for w in wrappers),
        })
        print(f"[✓] {k.__name__:30s}  -> {len(ptx):6d} B")

    # 2) update kernels.yaml
    yml = Path("/vol/kernels.yaml")
    old = yaml.safe_load(yml.read_text()) if yml.exists() else []
    yml.write_text(yaml.safe_dump(old + meta_rows, sort_keys=False))

    VOL.commit()
    print("\nSaved PTX + metadata to Modal volume:", VOL_NAME)
# ──────────────── CLI entrypoint ─────────────────────────────────────────
@APP.local_entrypoint()
def main(code_path: str, sm: int = 90):
    dump.remote(code_path, sm)
