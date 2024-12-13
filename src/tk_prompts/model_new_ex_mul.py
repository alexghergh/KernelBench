# import compiled ThunderKitten kernels
import tk_kernels # compiled kernel module would also be named tk_kernels


import torch
import torch.nn as nn
import torch.nn.functional as F

B = 1
H = 4
N = 16
D = 16

INPUT_DTYPE = torch.bfloat16
OUTPUT_DYPE = torch.float32


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        """
        o = x @ x.T
        input: x, y is bfloat16
        output: o is float32 (output accumulator)
        """
        output = torch.zeros(B, H, N, N, dtype=OUTPUT_DYPE)
        tk_kernels.dispatch_micro(x, y, output)

        return output
