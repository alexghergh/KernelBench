import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Perform matrix multiplication: a (128,128) x b (128,128) = (128,128)
        return torch.matmul(a.to(torch.half), b.to(torch.half)).float()

def get_inputs():
    # Generate input tensors for the model:
    # 'a' has shape (128, 128) and 'b' has shape (128, 128) so that torch.matmul(a, b) is valid.
    a = torch.randn(128, 128).cuda()
    b = torch.randn(128, 128).cuda()
    return [a, b]

def get_init_inputs():
    # If any initialization inputs are needed, add them here.
    return []