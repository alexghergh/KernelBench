import torch
import torch.nn as nn
import torch.nn.functional as F

B = 1 
N = 16
D = 32
DTYPE = torch.float32


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        # o = x - x 
        return x - x


def get_inputs():
    # randomly generate input tensors based on the model architecture
    x = torch.randn(B, N, D, dtype=DTYPE).cuda()
    return [x]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
