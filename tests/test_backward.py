import torch
import torch.nn as nn
import KernelBench.level1.backward.max_pooling_1d_bwd as max_pooling_1d_bwd
import KernelBench.level1.backward.leakyrelu_bwd as leakyrelu_bwd
import KernelBench.level1.backward.relu_bwd as relu_bwd
import KernelBench.level1.backward.conv_standard_2D__asymmetric_input__square_kernel__input_bwd as conv_55_input_bwd
import KernelBench.level1.backward.conv_standard_2D__asymmetric_input__square_kernel__weight_bwd as conv_55_weight_bwd

def test_max_pooling_1d_bwd():
    batch_size = 16
    features = 64
    sequence_length = 128
    kernel_size = 4
    stride = 2
    padding = 2
    dilation = 3
    return_indices = True
    # Generate input tensor with requires_grad=True
    x = torch.randn(batch_size, features, sequence_length, requires_grad=True)

    # Define MaxPool1d layer
    maxpool = torch.nn.MaxPool1d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices
    )

    # Run forward pass
    output, indices = maxpool(x)

    output.retain_grad()

    # Compute scalar loss
    loss = output.sum()

    # Run backward pass using autograd
    loss.backward()
    # Get the gradient of the output tensor
    grad_output_autograd = output.grad.clone()
    grad_x_autograd = x.grad.clone()

    bwd_model = max_pooling_1d_bwd.Model()
    grad_x = bwd_model(indices, grad_output_autograd, torch.zeros_like(x))

    # Compare results
    torch.testing.assert_close(grad_x, grad_x_autograd)

def test_leakyrelu_bwd():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim, requires_grad=True)
    negative_slope = 0.01
    # Define LeakyReLU layer
    leakyrelu = nn.LeakyReLU(negative_slope=negative_slope)
    output = leakyrelu(x)
    output.retain_grad()
    # Compute scalar loss
    loss = output.sum()
    # Run backward pass using autograd
    loss.backward()
    # Get the gradient of the output tensor
    grad_output_autograd = output.grad.clone()
    grad_x_autograd = x.grad.clone()

    bwd_model = leakyrelu_bwd.Model(negative_slope=negative_slope)
    grad_x = bwd_model(x, grad_output_autograd)

    # Compare results
    torch.testing.assert_close(grad_x, grad_x_autograd)

def test_relu_bwd():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim, requires_grad=True)
    # Define ReLU layer
    relu = nn.ReLU()
    output = relu(x)
    output.retain_grad()
    # Compute scalar loss
    loss = output.sum()
    # Run backward pass using autograd
    loss.backward()
    # Get the gradient of the output tensor
    grad_output_autograd = output.grad.clone()
    grad_x_autograd = x.grad.clone()

    bwd_model = relu_bwd.Model()
    grad_x = bwd_model(x, grad_output_autograd)

    # Compare results
    torch.testing.assert_close(grad_x, grad_x_autograd)

def test_conv_55_bwd():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    width = 256
    height = 128  # Asymmetric input
    # set the seed
    torch.manual_seed(0)

    # Generate input tensor with requires_grad=True
    x = torch.rand(batch_size, in_channels, height, width, requires_grad=True)

    # Define Conv2d layer
    conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size))

    # Run forward pass
    output = conv2d(x)

    output.retain_grad()

    # Compute scalar loss
    loss = output.sum()

    # Run backward pass using autograd
    loss.backward()
    # Get the gradient of the output tensor
    grad_output_autograd = output.grad.clone()
    grad_x_autograd = x.grad.clone()
    grad_weight_autograd = conv2d.weight.grad.clone()

    weight = conv2d.weight

    input_bwd_model = conv_55_input_bwd.Model()
    grad_x = input_bwd_model(weight, grad_output_autograd)

    # Compare results
    torch.testing.assert_close(grad_x, grad_x_autograd)

    weight_bwd_model = conv_55_weight_bwd.Model(kernel_size)
    grad_weight = weight_bwd_model(x, grad_output_autograd)

    # Compare results
    torch.testing.assert_close(grad_weight, grad_weight_autograd)

