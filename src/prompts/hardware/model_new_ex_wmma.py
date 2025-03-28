import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for wmma matmul
# https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 128
#define MATRIX_N 128
#define MATRIX_K 128

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_matmul_kernel(const half* a, const half* b, float* out, int M, int N, int K) {

    // Leading dimensions. Packed with no transpositions.
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> out_frag;

    wmma::fill_fragment(out_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;

    int bRow = i;
    int bCol = warpN * WMMA_N;

    // Bounds checking
    if (aRow < M && aCol < K && bRow < K && bCol < N) {
        // Load the inputs
        wmma::load_matrix_sync(a_frag, a + aRow * lda + aCol, lda);
        wmma::load_matrix_sync(b_frag, b + bRow * ldb + bCol, ldb);

        // Perform the matrix multiplication
        wmma::mma_sync(out_frag, a_frag, b_frag, out_frag);

    }
    }

    // Store the output
    int outRow = warpM * WMMA_M;
    int outCol = warpN * WMMA_N;

    if (outRow < M && outCol < N) {
        wmma::store_matrix_sync(out + outRow * ldc + outCol, out_frag, ldc, wmma::mem_row_major);
    }
}

torch::Tensor wmma_matmul_cuda(torch::Tensor a, torch::Tensor b) {

    torch::Tensor a_fp16 = a.to(torch::kHalf);
    torch::Tensor b_fp16 = b.to(torch::kHalf);
    auto out = torch::zeros_like(a);

    // First: using WMMA
    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    wmma_matmul_kernel<<<gridDim, blockDim>>>(
        reinterpret_cast<const __half*>(a_fp16.data_ptr<at::Half>()), 
        reinterpret_cast<const __half*>(b_fp16.data_ptr<at::Half>()), 
        out.data_ptr<float>(), MATRIX_M, MATRIX_N, MATRIX_K);

    return out;
}
"""

cpp_src = (
    "torch::Tensor wmma_matmul_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
wmma_matmul = load_inline(
    name="wmma_matmul",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["wmma_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wmma_matmul = wmma_matmul

    def forward(self, a, b):
        return self.wmma_matmul.wmma_matmul_cuda(a, b)