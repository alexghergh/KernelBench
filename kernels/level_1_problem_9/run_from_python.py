import tk_kernels
import torch
import torch.nn as nn

INPUT_DTYPE = torch.bfloat16
OUTPUT_DTYPE = torch.float

M = 64
N = 16

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        # Initialize output tensor
        output = torch.zeros(M, M, dtype=OUTPUT_DTYPE, device='cuda')
        
        # Move inputs to GPU if not already there
        A = A.cuda()
        B = B.cuda()
        
        # Call our custom kernel
        tk_kernels.dispatch_tall_matmul(A, B, output)
        
        return output
    

A = torch.ones(M, N, dtype=INPUT_DTYPE)
B = torch.ones(N, M, dtype=INPUT_DTYPE)

output_ref = torch.matmul(A, B).to(OUTPUT_DTYPE)
print(output_ref.shape)
print(output_ref.mean())

model = ModelNew().cuda()
output = model(A, B)
print(output.shape)
print(output.mean())