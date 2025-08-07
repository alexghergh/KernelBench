# KernelBench/ptx_runner.py
import json
import cupy as cp
import torch
import torch.nn as nn

_DTYPE_MAP = {
    "torch.float16": cp.float16,
    "torch.float32": cp.float32,
    "torch.float64": cp.float64,
    "torch.int32":   cp.int32,
    "torch.int64":   cp.int64,
}

def _torch_to_cu(t: torch.Tensor) -> cp.ndarray:
    """Zero-copy bridge Torch->CuPy via DLPack."""
    return cp.fromDlpack(torch.utils.dlpack.to_dlpack(t))

def _ensure_inputs_on_cuda(*xs):
    out = []
    for x in xs:
        if isinstance(x, torch.Tensor):
            out.append(x.to(device="cuda", non_blocking=True))
        else:
            out.append(x)
    return out

class PTXKernel(nn.Module):
    """
    Minimal wrapper to launch a single PTX kernel described by a manifest.
    Assumes your JSON follows the schema we wrote (args order, tensors metadata, grid/block, etc).
    """
    def __init__(self, ptx_path: str, json_path: str):
        super().__init__()
        with open(json_path) as f:
            man = json.load(f)
        self.man = man
        with open(ptx_path, "r") as f:
            ptx = f.read()
        self.mod = cp.RawModule(code=ptx, backend="nvrtc")
        self.func = self.mod.get_function(man["kernel_name"])

        self.grid  = tuple(man["grid"])
        self.block = tuple(man["block"])
        self.args_spec = man["args"]           # ["A:ptr","B:ptr","C:ptr","M:u32",...]
        self.scalars   = man.get("scalars", {})
        self.tensors   = man.get("tensors", {})

    def forward(self, *inputs: torch.Tensor):
        # Example: for matmul we expect (A,B) and we’ll allocate C like reference did.
        inputs = _ensure_inputs_on_cuda(*inputs)
        # Map *runtime* tensors to args: we’ll replace manifest tensor entries (A,B,C,…) with actual buffers.
        # Heuristic: keys present in manifest["tensors"] that are not provided in inputs will be allocated here.
        name_to_cp = {}

        # Bind provided inputs first by common names if possible (A,B,...) else by order of ptr args encountered
        ptr_names = [a.split(":")[0] for a in self.args_spec if a.endswith(":ptr")]
        provided = [x for x in inputs if isinstance(x, torch.Tensor)]
        for name, t in zip(ptr_names, provided):
            name_to_cp[name] = _torch_to_cu(t.contiguous())

        # Allocate any remaining ptr buffers from manifest metadata (e.g., C)
        for name in ptr_names:
            if name in name_to_cp:
                continue
            meta = self.tensors.get(name)
            if not meta:
                raise RuntimeError(f"No tensor meta for required arg '{name}'.")
            shape  = tuple(meta["shape"])
            dt_key = meta["dtype"]
            dtype  = _DTYPE_MAP.get(dt_key)
            if dtype is None:
                raise RuntimeError(f"Unsupported dtype in manifest: {dt_key}")
            name_to_cp[name] = cp.empty(shape, dtype=dtype)

        # Build the raw arg list in the EXACT order specified
        raw_args = []
        for spec in self.args_spec:
            name, kind = spec.split(":")
            if kind == "ptr":
                raw_args.append(name_to_cp[name])
            else:
                # cupy will pack ints/floats properly if passed as python scalars
                raw_args.append(self.scalars[name])

        # Launch
        self.func(
            grid=self.grid,
            block=self.block,
            args=tuple(raw_args),
            shared_mem=0
        )
        # Return the first output-looking tensor; here we assume "C" if present else last ptr
        out_name = "C" if "C" in name_to_cp else ptr_names[-1]
        # Convert back to torch (still zero-copy)
        return torch.utils.dlpack.from_dlpack(name_to_cp[out_name].toDlpack())
