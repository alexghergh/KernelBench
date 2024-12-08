# import compiled ThunderKitten kernels
import tk_kernels # compiled kernel module would also be named tk_kernels


import torch
import torch.nn as nn
import torch.nn.functional as F

B = 1 
N = 16
D = 32
DTYPE = torch.float32


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        output = torch.zeros_like(x, dtype=DTYPE)
        tk_kernels.dispatch_micro(x, output)
        return output
