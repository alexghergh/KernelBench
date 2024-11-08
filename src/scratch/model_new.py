import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for diagonal matrix multiplication
diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* A, const float* B, float* C, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / M;
    int j = idx % M;
    if (i < N && j < M) {
        C[idx] = A[i * M + i] * B[i * M + j];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto M = B.size(1);
    auto C = torch::zeros({N, M}, B.options());

    const int block_size = 256;
    const int num_blocks = (N * M + block_size - 1) / block_size;

    diag_matmul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);

    return C;
}
"""

diag_matmul_cpp_source = "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for diagonal matrix multiplication
diag_matmul = load_inline(
    name='diag_matmul',
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_source,
    functions=['diag_matmul_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul = diag_matmul
    
    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A, B)