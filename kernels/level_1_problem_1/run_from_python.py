import tk_kernels
import torch

input = torch.ones(16, 1024, 32, 64, device='cuda')
output = torch.zeros_like(input)
tk_kernels.copy_kernel(input, output)
print(output.mean(), '\n')

output = torch.zeros_like(input)
tk_kernels.wrapped_copy_kernel(input, output)
print(output.mean())