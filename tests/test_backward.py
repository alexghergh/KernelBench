import torch
import KernelBench.level1.backward.max_pooling_1d_bwd as max_pooling_1d_bwd

def test_backward():
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
