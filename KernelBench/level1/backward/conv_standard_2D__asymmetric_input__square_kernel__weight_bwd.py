import torch
import torch.nn as nn
import torch.nn.functional as F

# Backward of weight for 55_conv_standard_2D__asymmetric_input__square_kernel.py

class Model(nn.Module):
    def __init__(self, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
        super(Model, self).__init__()
        self.kernel_h = kernel_size
        self.kernel_w = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input: torch.Tensor, grad_output: torch.Tensor):
        """
        Computes the gradient with respect to the weight of a 2D convolution operation.

        Args:
            input (torch.Tensor): Input tensor of the convolution layer.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output of the convolution layer.

        Returns:
            torch.Tensor: Gradient of the loss with respect to the weight of the convolution layer.
        """
        N, C_in, _, _ = input.shape
        N, C_out, _, _ = grad_output.shape

        # Extract patches from input
        input_patches = F.unfold(
            input,
            kernel_size=(self.kernel_h, self.kernel_w),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )  # Shape: [N, C_in * K_h * K_w, L], where L = H_out * W_out

        # Reshape grad_output to [N, C_out, L]
        grad_output_reshaped = grad_output.view(N, C_out, -1)  # Shape: [N, C_out, L]

        # Compute grad_weight using batch matrix multiplication
        # Transpose input_patches to [N, L, C_in * K_h * K_w]
        input_patches_transposed = input_patches.transpose(1, 2)  # Shape: [N, L, C_in * K_h * K_w]

        # Batch matrix multiplication and sum over batch dimension
        grad_weight = torch.bmm(
            grad_output_reshaped,  # [N, C_out, L]
            input_patches_transposed  # [N, L, C_in * K_h * K_w]
        )  # Resulting shape: [N, C_out, C_in * K_h * K_w]

        # Sum over the batch dimension
        grad_weight = grad_weight.sum(dim=0)  # Shape: [C_out, C_in * K_h * K_w]

        # Reshape to [C_out, C_in, K_h, K_w]
        grad_weight = grad_weight.view(C_out, C_in, self.kernel_h, self.kernel_w)

        # Adjust for groups if necessary
        if self.groups > 1:
            grad_weight = grad_weight.view(self.groups, C_out // self.groups, C_in // self.groups, self.kernel_h, self.kernel_w)
            grad_weight = grad_weight.reshape(C_out, C_in // self.groups, self.kernel_h, self.kernel_w)

        return grad_weight
    
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
    return [x, grad_output]

def get_init_inputs():
    return [kernel_size]  # Provide kernel_size for initialization
