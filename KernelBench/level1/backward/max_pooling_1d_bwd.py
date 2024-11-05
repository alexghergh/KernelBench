import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that performs the backward of Max Pooling 1D.
    """
    def __init__(self):
        """
        Initializes the Max Pooling 1D layer for backward.
        """

        super(Model, self).__init__()

    def forward(self, indices: torch.Tensor, grad_output: torch.Tensor, grad_input: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D backward pass to the gradient of output.

        Args:
            indices (torch.Tensor): Indices tensor of shape (batch_size, num_features, output_sequence_length).
            grad_output (torch.Tensor): Output gradient tensor of shape (batch_size, num_features, output_sequence_length).
            grad_input (torch.Tensor): Input gradient tensor of shape (batch_size, num_features, original_sequence_length) initialized with zeros.

        Returns:
            torch.Tensor: Input gradient tensor, shape (batch_size, num_features, original_sequence_length).
        """
        input_size = grad_input.size()
        # Reshape tensors for scattering
        grad_input = grad_input.view(grad_input.size(0), grad_input.size(1), -1)
        indices = indices.view(indices.size(0), indices.size(1), -1)
        grad_output = grad_output.view(grad_output.size(0), grad_output.size(1), -1)
        # Scatter grad_output into grad_input using indices
        grad_input.scatter_add_(2, indices, grad_output)
        # Reshape grad_input to original input size
        grad_input = grad_input.view(input_size)
        return grad_input

batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, features, sequence_length)
    output, indices = F.max_pool1d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    grad_output = torch.randn_like(output)
    grad_input = torch.zeros_like(x)
    return [indices, grad_output, grad_input]

def get_init_inputs():
    return []