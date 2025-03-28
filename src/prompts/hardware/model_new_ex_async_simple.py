import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for async_memcpy
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/#without-memcpy-async
source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    #include <cooperative_groups.h>

    using namespace cooperative_groups;

    __global__ void memcpy_async_simple(float* global_out, float const* global_in, size_t size, size_t batch_sz) {
        // For simplicity, use built-in indices instead of cooperative group indices if not launched cooperatively.
        auto grid = cooperative_groups::this_grid();
        auto block = cooperative_groups::this_thread_block();
        assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

        // Allocate shared memory as floats.
        extern __shared__ float shared[];

        size_t local_idx = block.thread_rank();  // Each thread's index within the block

        for (size_t batch = 0; batch < batch_sz; ++batch) {
            // Compute global index for this batch element
            size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
            size_t global_idx = block_batch_idx + local_idx;
            shared[local_idx] = global_in[global_idx];

            block.sync(); // Waits for all threads to finish writing to shared memory

            if (global_idx < size) {
                global_out[global_idx] = shared[local_idx] * 2;
            }
            
            block.sync(); // Waits for all threads to finish writing to global memory
        }
    }

    torch::Tensor async_cuda(torch::Tensor a) {
        auto out = torch::zeros_like(a);
        int size = a.numel();
        int batch_sz = 4;

        float* global_out = out.data_ptr<float>();
        float* global_in = a.data_ptr<float>();

        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (size + (blockDim.x * batch_sz - 1)) / (blockDim.x * batch_sz);

        // Calculate required shared memory size (one float per thread)
        size_t sharedMemSize = blockDim.x * sizeof(float);

        memcpy_async_simple<<<gridDim, blockDim, sharedMemSize>>>(global_out, global_in, size, batch_sz);

        return out;
    }
"""

cpp_src = (
    "torch::Tensor async_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for element-wise addition
async_cuda = load_inline(
    name="async_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["async_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.async_cuda = async_cuda

    def forward(self, a):
        return self.async_cuda.async_cuda(a)