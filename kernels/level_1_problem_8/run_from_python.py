import tk_kernels
import torch
import torch.nn as nn

INPUT_DTYPE = torch.bfloat16
OUTPUT_DTYPE = torch.float

M = 16
N = 16384
K = N

class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        output = torch.zeros(M, M, dtype=OUTPUT_DTYPE, device='cuda')

        A = A.cuda()
        B = B.cuda()
        tk_kernels.dispatch_micro(A, B, output, K)
        
        return output


A = torch.rand(M, N, dtype=INPUT_DTYPE) / N # [16, 1024]
B = torch.rand(N, M, dtype=INPUT_DTYPE) / N # [1024, 16]

A = A.cuda()
B = B.cuda()

output_ref = torch.matmul(A, B).to(OUTPUT_DTYPE)
print("Ref output shape:", output_ref.shape)
print("Ref output mean:", output_ref.mean())


model = ModelNew().cuda()
output = model(A, B)
print("TK Output shape:", output.shape)
print("TK Output mean:", output.mean())

# import pdb; pdb.set_trace()

assert torch.allclose(output, output_ref, atol=1e-2)