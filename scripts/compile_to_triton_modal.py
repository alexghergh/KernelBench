"""
compile_to_triton_modal.py
==========================
Compile a PyTorch "problem" with torch.compile() on a Modal GPU,
forcing Triton-only kernel generation and persisting all artifacts
(including the final Triton @jit kernels) into a Modal Volume.

Examples (WSL/bash):
  # Pick a problem by ID from a folder
  modal run compile_to_triton_modal.py::main \
    --problems_dir "/mnt/c/Users/.../KernelBench/KernelBench/level2" \
    --pick_id 1 \
    --gpu H100

  # Direct file
  modal run compile_to_triton_modal.py::main \
    --code_file ../KernelBench/level2/3_Batched_matrix_multiplication.py \
    --gpu H100

  # Paste from stdin
  modal run compile_to_triton_modal.py::main --code_file - --gpu H100 <<'PY'
  import torch, torch.nn as nn
  class Tiny(nn.Module):
      def __init__(self,d=1024): super().__init__(); self.l1=nn.Linear(d,4*d); self.l2=nn.Linear(4*d,d)
      def forward(self,x): return self.l2(torch.nn.functional.gelu(self.l1(x)))
  def make_problem(): return Tiny(1024), (torch.randn(4096,1024,dtype=torch.float16),)
  PY
"""

import os
import pathlib
import modal

# NEW: Set TMPDIR inside the container. This is where torchinductor_* will be created.
os.environ["TMPDIR"] = "/vol/tmp"

# -------------------------
# Build Modal image
# -------------------------
REPO_TOP_DIR = os.path.dirname(os.path.abspath(__file__))
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    # CORRECTED: Use a single, reliable source for both torch and triton nightlies.
    .run_commands(
        "python -m pip install --upgrade pip",
        # Install both torch and a compatible triton from the official PyTorch nightly index
        "python -m pip install --pre torch triton --index-url https://download.pytorch.org/whl/nightly/cu121",
        # Install other packages separately
        "python -m pip install numpy tqdm python-dotenv",
    )
    .env({"FORCE_REBUILD_V2": "1"})
)

# -------------------------
# Persisted volume to keep artifacts between runs
# -------------------------
VOLUME_NAME = "triton-kernel-dumps"
try:
    # Newer Modal SDK
    vol = modal.Volume.persisted(VOLUME_NAME)
except AttributeError:
    # Older SDKs
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App("compile-to-triton-v2")

# -------------------------
# Remote worker
# -------------------------
@app.cls(image=image, volumes={"/vol": vol}, gpu="H100")
class InductorDumper:
    @modal.method()
    def compile_and_dump(
        self,
        module: str | None = None,
        symbol: str = "make_problem",
        code: str | None = None,
        mode: str = "max-autotune",
        device: str = "cuda",
        dump_triton_console: bool = False,
        force_triton_only: bool = True,
        save_to_volume: bool = True,
    ):
        """
        Returns JSON with:
          - verification: Summary of non-Triton calls found.
          - run_dir:      Volume path where this run's artifacts were saved.
          - archive:      Volume path to a tar.gz of the run.
          - triton_index: {file -> [kernel_function_names]} from async_compile.
        """
        import sys, tempfile, importlib, json, inspect, shutil, tarfile, time
        import torch

        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        os.environ.setdefault("TORCH_LOGS", "output_code")
        os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")
        os.environ.setdefault("TRITON_KERNEL_DUMP", "1")
        triton_cache_path = pathlib.Path(f"/vol/tmp/triton_cache_{int(time.time())}")
        os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache_path))
        triton_cache_path.mkdir(parents=True, exist_ok=True)

        if force_triton_only:
            print("\n=== Forcing Triton-only compilation mode ===")
            try:
                import torch._inductor.config as ind_cfg

                if hasattr(ind_cfg, "autotune_fallback_to_aten"):
                    print(" - Setting autotune_fallback_to_aten = False")
                    ind_cfg.autotune_fallback_to_aten = False

                if hasattr(ind_cfg, "use_aten_gemm_kernels"):
                    print(" - Setting use_aten_gemm_kernels = False")
                    ind_cfg.use_aten_gemm_kernels = False

                # CORRECTED: Use a string, not a list, for the backend values.
                if hasattr(ind_cfg, "max_autotune_gemm_backends"):
                    print(" - Restricting max_autotune_gemm_backends to 'TRITON'")
                    ind_cfg.max_autotune_gemm_backends = "TRITON"
                if hasattr(ind_cfg, "max_autotune_conv_backends"):
                    print(" - Restricting max_autotune_conv_backends to 'TRITON'")
                    ind_cfg.max_autotune_conv_backends = "TRITON"
                
                if hasattr(ind_cfg, "use_triton_template"):
                     print(" - Setting use_triton_template = True")
                     ind_cfg.use_triton_template = True

                if hasattr(ind_cfg, "triton"):
                    ind_cfg.triton.unique_kernel_names = True

            except ImportError as e:
                print(f"(warn) Could not import torch._inductor.config: {e}")
            except Exception as e:
                print(f"(warn) Unable to set some inductor config flags: {e}")

            try:
                print(" - Disabling torch.backends.cudnn")
                torch.backends.cudnn.enabled = False
                print(" - Disabling fused SDP backends")
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception as e:
                print(f"(warn) Unable to disable some backends: {e}")
        
        # ... (the rest of the function remains the same) ...
        if code:
            tmpdir = tempfile.mkdtemp(dir="/vol/tmp")
            modpath = pathlib.Path(tmpdir) / "user_problem.py"
            modpath.write_text(code)
            sys.path.insert(0, tmpdir)
            module = "user_problem"

        if not module:
            raise RuntimeError("Provide either 'module' or 'code'.")

        mod = importlib.import_module(module)
        if hasattr(mod, symbol): make_problem = getattr(mod, symbol)
        else:
            def _kb_make_problem():
                init_inputs = mod.get_init_inputs()
                model = mod.Model(*init_inputs)
                run_inputs = mod.get_inputs()
                return model.eval(), run_inputs
            make_problem = _kb_make_problem

        fn, example_inputs = make_problem()
        def _to_dev(x):
            if isinstance(x, torch.Tensor): return x.to(device)
            if isinstance(x, (list, tuple)): return type(x)(_to_dev(xx) for xx in x)
            if isinstance(x, dict): return {k: _to_dev(v) for k, v in x.items()}
            return x
        example_inputs = _to_dev(example_inputs)
        if isinstance(fn, torch.nn.Module): fn = fn.to(device).eval()

        print(f"\n=== Compiling with mode='{mode}' ===")
        compiled = torch.compile(fn, mode=mode, backend="inductor")
        with torch.no_grad():
            _ = compiled(*example_inputs)

        debug_root = pathlib.Path(".") / "torch_compile_debug"
        latest_debug_dir = sorted(debug_root.glob("*"))[-1] if debug_root.exists() else None
        
        verification_results = {}
        if latest_debug_dir:
            verification_results = self._verify_full_triton(latest_debug_dir)

        async_dirs = self._collect_async_compile(only_latest_n=1)
        triton_index = {}
        if async_dirs:
            latest_async_dir = async_dirs[0]
            print(f"\n=== Found Triton kernels in: {latest_async_dir} ===")
            if dump_triton_console:
                triton_index = self._dump_triton_kernels_from_dir(latest_async_dir)
        else:
            print("\n(warn) No 'async_compile' directory found. Triton kernels may not have been generated.")

        run_dir_vol, archive_vol = None, None
        if save_to_volume:
            ts = time.strftime("%Y%m%d-%H%M%S")
            run_dir = pathlib.Path("/vol") / "runs" / ts
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving artifacts to: {run_dir}")

            if async_dirs:
                ac_out = run_dir / "triton_kernels"
                shutil.copytree(async_dirs[0], ac_out, dirs_exist_ok=True)
                print(f"  - Copied Triton kernels to: {ac_out}")
                triton_index = self._dump_triton_kernels_from_dir(ac_out)

            if latest_debug_dir:
                dest = run_dir / "torch_compile_debug"
                shutil.copytree(latest_debug_dir, dest, dirs_exist_ok=True)
                print(f"  - Copied debug logs to: {dest}")
            
            dest_cache = run_dir / "triton_cache"
            shutil.copytree(triton_cache_path, dest_cache, dirs_exist_ok=True)
            print(f"  - Copied Triton cache to: {dest_cache}")

            archives_dir = pathlib.Path("/vol") / "archives"
            archives_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archives_dir / f"{ts}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(run_dir, arcname=f"run_{ts}")

            run_dir_vol = str(run_dir)
            archive_vol = str(archive_path)
            print(f"\nSaved artifacts to volume:\n  run_dir = {run_dir_vol}\n  archive = {archive_vol}")

        return {
            "verification": verification_results,
            "run_dir": run_dir_vol,
            "archive": archive_vol,
            "triton_index": triton_index,
            "volume_name": VOLUME_NAME if save_to_volume else None,
        }
    # NEW: Helper to find the async_compile directory
    @staticmethod
    def _collect_async_compile(only_latest_n: int = 1):
        import glob
        tmpdir = os.environ.get("TMPDIR", "/tmp")
        # Path: /tmp/torchinductor_<user>/<hash>/async_compile
        pattern = os.path.join(tmpdir, "torchinductor_*", "*", "async_compile")
        roots = glob.glob(pattern)
        paths = [pathlib.Path(p) for p in roots if pathlib.Path(p).is_dir()]
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return paths[:only_latest_n]

    # NEW: Verification function to check for non-Triton fallbacks
    @staticmethod
    def _verify_full_triton(latest_debug_dir: "pathlib.Path") -> dict:
        import re
        output_code_file = latest_debug_dir / "output_code.py"
        summary = {"checked_file": str(output_code_file), "extern_hits": 0, "aten_hits": 0, "status": "UNKNOWN"}
        
        print("\n=== Full-Triton Verification ===")
        if not output_code_file.exists():
            print(f" - SKIPPED: {output_code_file} not found.")
            summary["status"] = "SKIPPED"
            return summary

        text = output_code_file.read_text()
        # Look for calls to aten operations that were not fused
        extern_hits = len(re.findall(r"\bextern_kernels\.", text))
        aten_hits = len(re.findall(r"\baten\.", text))
        
        summary["extern_hits"] = extern_hits
        summary["aten_hits"] = aten_hits

        print(f" - Scanned: {output_code_file}")
        print(f" - extern_kernels.* calls: {extern_hits}")
        print(f" - aten.* calls: {aten_hits}")
        
        if extern_hits == 0 and aten_hits == 0:
            print(" - Status: SUCCESS - No obvious non-Triton calls found.")
            summary["status"] = "SUCCESS"
        else:
            print(" - Status: FAILURE - Non-Triton calls detected. Check output_code.py.")
            summary["status"] = "FAILURE"
            
        return summary

    # UPDATED: This function is now more of a general utility
    @staticmethod
    def _dump_triton_kernels_from_dir(directory: "pathlib.Path", print_to_console: bool = True):
        import re, os
        index = {}
        py_files = list(directory.rglob("*.py"))
        if not py_files:
            if print_to_console: print(f"(No .py files found in {directory})")
            return index

        total_kernels = 0
        for py in py_files:
            text = py.read_text()
            pattern = re.compile(r"(@triton\.jit\s*def\s+\w+\(.*\):)")
            matches = list(pattern.finditer(text))
            if not matches: continue
            
            rel_path = os.path.relpath(py, directory)
            index[rel_path] = [m.group(1) for m in matches]
            total_kernels += len(matches)

            if print_to_console:
                print(f"\n--- Kernels in: {rel_path} ---")
                print(text)
        
        if print_to_console:
            print(f"\n=== Found {total_kernels} @triton.jit kernels in {len(index)} files ===")
        return index


# -------------------------
# Local entrypoint (no changes needed)
# -------------------------
@app.local_entrypoint()
def main(
    # ... your local entrypoint arguments are fine ...
    module: str = "",
    symbol: str = "make_problem",
    code_file: str = "",
    code: str = "",
    paste: bool = False,
    mode: str = "max-autotune",
    device: str = "cuda",
    gpu: str = "H100",
    problems_dir: str = "../KernelBench/level2",
    pick_id: int = 0,
    name_substr: str = "",
    glob_pat: str = "*.py",
    dump_triton_console: bool = True, # CHANGED: Default to True to see kernels
    save_to_volume: bool = True,
    list_only: bool = False,
    strict_id: bool = False,
):
    import sys, json, pathlib, re, shutil, subprocess

    base_dir_arg = problems_dir
    if os.name == "posix" and re.match(r"^[A-Za-z]:[\\/]", base_dir_arg or "") and shutil.which("wslpath"):
        try:
            out = subprocess.check_output(["wslpath", "-u", base_dir_arg], text=True).strip()
            problems_dir = out
        except Exception: pass
    base_dir = (pathlib.Path(__file__).parent / problems_dir).resolve()
    src = None
    if code_file == "-": src = sys.stdin.read()
    elif code_file: src = pathlib.Path(code_file).read_text()
    elif paste:
        print("Paste your code, then Ctrl-D (Unix/macOS) or Ctrl-Z+Enter (Windows):")
        src = sys.stdin.read()
    elif code: src = code
    elif module: src = None
    else:
        if not base_dir.exists(): raise SystemExit(f"Problems dir not found: {base_dir}")
        files = sorted(p for p in base_dir.glob(glob_pat) if p.is_file())
        if not files: raise SystemExit(f"No files matched {glob_pat} under {base_dir}")
        if list_only:
            print(f"Listing problems under {base_dir}:")
            for p in files: print(f"  {p.name}")
            return
        chosen = None
        if name_substr:
            candidates = [p for p in files if name_substr.lower() in p.name.lower()]
            if not candidates: raise SystemExit(f"No file name contains '{name_substr}'")
            chosen = candidates[0]
        elif pick_id:
            num_prefix = re.compile(rf"^{pick_id}[_\-].*\.py$", re.IGNORECASE)
            by_prefix = [p for p in files if num_prefix.match(p.name)]
            if by_prefix: chosen = by_prefix[0]
            elif not strict_id and (1 <= pick_id <= len(files)): chosen = files[pick_id - 1]
            else: raise SystemExit(f"Could not find problem for --pick_id {pick_id}")
        else:
            print(f"Found {len(files)} files. Pass --pick_id N or --name_substr 'text' to choose.")
            return
        print(f"[local] Selected:\n{chosen}")
        src = chosen.read_text()
    if (module and src) or (not module and not src):
        raise SystemExit("Provide exactly one of: --module OR code source (file/pick/paste).")

    modal.enable_output()
    # Note: Added `force_triton_only=True` to the call
    res = InductorDumper.with_options(gpu=gpu)().compile_and_dump.remote(
        module=module or None,
        symbol=symbol,
        code=src,
        mode=mode,
        device=device,
        dump_triton_console=dump_triton_console,
        force_triton_only=True, # Explicitly pass the flag
        save_to_volume=save_to_volume,
    )
    print("\n--- FINAL RESULT ---")
    print(json.dumps(res, indent=2))