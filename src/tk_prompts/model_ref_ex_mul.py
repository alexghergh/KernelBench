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
        o = torch.matmul(x, y.transpose(-2, -1)).to(OUTPUT_DYPE)
        return o

def get_inputs():
    # randomly generate input tensors based on the model architecture
    x = torch.randn(B, H, N, D, dtype=INPUT_DTYPE).cuda()
    y = torch.randn(B, H, N, D, dtype=INPUT_DTYPE).cuda()
    return [x, y]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
