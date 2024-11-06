import torch
import torch.nn as nn
import torch.nn.functional as F

# Backward of input for 55_conv_standard_2D__asymmetric_input__square_kernel.py

class Model(nn.Module):
    def __init__(self, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
        super(Model, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, weight: torch.Tensor, grad_output: torch.Tensor):
        """
        Computes the gradient with respect to the input of a 2D convolution operation.

        Args:
            weight (torch.Tensor): Weight tensor of the convolution layer.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output of the convolution layer.

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input of the convolution layer.
        """
        grad_input = F.conv_transpose2d(
            grad_output,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=0,
            groups=self.groups,
            dilation=self.dilation
        )
        
        return grad_input
    
# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 128  # Asymmetric input

def get_inputs():
    conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size))
    x = torch.randn(batch_size, in_channels, height, width)
    output = conv2d(x)
    grad_output = torch.randn_like(output)
    # Get the weight of the convolution layer
    weight = conv2d.weight
    return [weight, grad_output]

def get_init_inputs():
    return []  # Provide stride, padding, dilation, groups for initialization