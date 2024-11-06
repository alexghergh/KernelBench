import torch
import torch.nn as nn

# Backward for level1/19_ReLU.py

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, input: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU gradient applied, same shape as input.
        """
        return torch.where(input > 0, 1.0, 0.0)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    output = torch.nn.functional.relu(x)
    grad_output = torch.randn_like(output)
    return [x, grad_output]

def get_init_inputs():
    return []  # No special initialization inputs needed