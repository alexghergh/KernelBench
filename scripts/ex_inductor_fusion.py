import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor_debug"
os.environ["TORCHINDUCTOR_DISABLE_CACHE_HIT"] = "1"

"""
Ideally we expect log to show explicit fusion decisions such as
[torch._inductor.fx_passes.fuse] fusing aten.linear, aten.relu → fused_0

Run with TORCH_LOGS=inductor python3 ex_inductor_fusion.py

Note: this doesn't work right now
"""

import torch
import torch.nn as nn
from torch._inductor import config
from torch._inductor.ir import IRNode
from torch._inductor.fx_passes.pre_grad import run as run_inductor_passes
from torch._inductor.graph import GraphLowering

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

model = MyModel()
inp = torch.randn(64, 128)

# Wrap with torch.compile
compiled = torch.compile(model, backend="inductor", fullgraph=True)
compiled(inp)  # First run triggers compile

# Now: dump the Inductor fusion IR (via torch.fx)
# Safe version: use export and inspect the lowered FX
exported = torch.export.export(model, (inp,))
print("\n=== FX Graph ===")
print(exported.graph_module.code)

# Bonus: If you're on nightly and want IR, dump it from inductor via hooks