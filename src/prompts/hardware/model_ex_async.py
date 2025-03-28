import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a):
        return a * 2

def get_inputs():
    # Generate input tensors for the model:
    a = torch.randn(16384, device="cuda", dtype=torch.float32)
    return [a]

def get_init_inputs():
    # If any initialization inputs are needed, add them here.
    return []