"""
compile_to_triton_modal.py
==========================
Compile a PyTorch "problem" with torch.compile(mode="max-autotune") on a Modal GPU,
print the generated Triton/C++ kernels (via TORCH_LOGS=output_code), and summarize
where debug artifacts were written.

Usage examples:
  modal run compile_to_triton_modal.py::main --code-file example_problem.py
  modal run compile_to_triton_modal.py::main --module user_problem --symbol make_problem --gpu H100
"""

import os
import pathlib
import modal

# ---------------------------------------------------------------------------
# Image: mirror the style of run_triton_generation.py
# ---------------------------------------------------------------------------

REPO_TOP_DIR = os.path.dirname(os.path.abspath(__file__))
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install(
        # pin to match your working environment
        "torch==2.5.0",
        "triton",            # Triton kernels on CUDA
        "numpy",
        "tqdm",
        "python-dotenv",
        # optional: match your base repo utils if needed
        # "pydra_config",
        # "pytest",
        # "ninja",
    )
    # This env flag pattern comes from your working script
    .env({"FORCE_REBUILD_V2": "1"})
)

app = modal.App("compile_to_tritonV2")


# Optional: map friendly GPU strings to Modal GPU specs elsewhere in your codebase.
# We keep the calling convention identical to your script: you'll pass a string
# to with_options(gpu=...), e.g. "H100", which your Modal account may already accept.
# If your account expects objects, replace that call site with modal.gpu.H100() etc.

@app.cls(image=image)
class InductorDumper:
    @modal.method()
    def compile_and_dump(
        self,
        module: str | None = None,         # import path containing make_problem()
        symbol: str = "make_problem",
        code: str | None = None,           # raw python source (alternative to module)
        mode: str = "max-autotune",
        device: str = "cuda",
        extra_logs: bool = True,
    ):
        """
        Returns: dict with 'debug_dirs' (last few dirs) and 'printed_notice'
        """
        import sys
        import tempfile
        import importlib
        import json
        import torch

        # Mirror your env logging behavior so kernels are printed to stdout.
        os.environ.setdefault("TORCH_LOGS", "output_code")
        os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")

        if extra_logs:
            try:
                import torch._inductor.config as ind_cfg
                ind_cfg.debug = True
            except Exception:
                pass

        # Accept either raw source or a module import path
        if code:
            tmpdir = tempfile.mkdtemp()
            modpath = pathlib.Path(tmpdir) / "user_problem.py"
            modpath.write_text(code)
            sys.path.insert(0, tmpdir)
            module = "user_problem"

        if not module:
            raise RuntimeError("Provide either 'module' (import path) or 'code' (source string).")

        mod = importlib.import_module(module)
        if not hasattr(mod, symbol):
            raise RuntimeError(
                f"Module {module!r} must define {symbol}() returning "
                "(callable_or_module, example_inputs[, example_kwargs])."
            )

        make_problem = getattr(mod, symbol)
        out = make_problem()
        if not (isinstance(out, tuple) and len(out) >= 2):
            raise RuntimeError(f"{symbol}() must return (callable, example_inputs[, example_kwargs]).")

        fn = out[0]
        example_inputs = out[1]
        example_kwargs = out[2] if len(out) >= 3 and isinstance(out[2], dict) else {}

        # Move tensors / nested structures to device
        def _to_dev(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, (list, tuple)):
                return type(x)(_to_dev(xx) for xx in x)
            if isinstance(x, dict):
                return {k: _to_dev(v) for k, v in x.items()}
            return x

        example_inputs = _to_dev(example_inputs)
        example_kwargs = _to_dev(example_kwargs)

        if isinstance(fn, torch.nn.Module):
            fn = fn.to(device).eval()

        compiled = torch.compile(fn, mode=mode, backend="inductor")

        # Trigger compile (and therefore kernel printouts) with one run
        with torch.no_grad():
            _ = compiled(*example_inputs, **example_kwargs)

        # Summarize where debug files were written
        debug_root = pathlib.Path(".") / "torch_compile_debug"
        debug_dirs = [str(p) for p in debug_root.glob("*") if p.is_dir()]
        debug_dirs.sort()
        tail = debug_dirs[-5:]

        notice = (
            "torch.compile finished. With TORCH_LOGS=output_code, generated kernels "
            "(Triton on CUDA, C++ on CPU) are printed above. "
            f"Found {len(debug_dirs)} debug dirs under ./torch_compile_debug/."
        )

        print("\n=== Summary ===")
        print(notice)
        for d in tail:
            print(f" - {d}")

        return {"debug_dirs": tail, "printed_notice": notice}


@app.local_entrypoint()
def main(
    module: str = "",            # e.g. "my_problem"
    symbol: str = "make_problem",
    code_file: str = "",         # e.g. "example_problem.py"
    mode: str = "max-autotune",
    device: str = "cuda",
    gpu: str = "H100",           # keep identical calling style to your script
):
    """
    CLI entrypoint (use with `modal run`).

    Examples:
      modal run compile_to_triton_modal.py::main --code-file example_problem.py
      modal run compile_to_triton_modal.py::main --module my_problem --symbol make_problem --gpu H100
    """
    import json

    src = None
    if code_file:
        src = pathlib.Path(code_file).read_text()

    # Match your pattern: enable Modal output and run the app context explicitly
    modal.enable_output()
    with app.run():
        # Keep identical 'with_options(gpu=gpu)' style as your working script.
        # If your Modal account requires objects, change to modal.gpu.H100() here.
        res = InductorDumper.with_options(gpu=gpu)().compile_and_dump.remote(
            module=module or None,
            symbol=symbol,
            code=src,
            mode=mode,
            device=device,
        )
        print("Result:", json.dumps(res, indent=2))
