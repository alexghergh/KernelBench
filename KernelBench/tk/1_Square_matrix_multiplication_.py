import torch
import torch.nn as nn

INPUT_DTYPE = torch.bfloat16
OUTPUT_DTYPE = torch.float

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B).to(OUTPUT_DTYPE)

N = 2048

def get_inputs():
    A = torch.randn(N, N, dtype=INPUT_DTYPE)
    B = torch.randn(N, N, dtype=INPUT_DTYPE)
    return [A, B]
    
def get_init_inputs():
    return []  # No special initialization inputs needed