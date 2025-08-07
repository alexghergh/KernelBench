#include <torch/extension.h>
#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>  // *** for current stream
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>   // add this


#define CUDA_CHECK(call)                                                    \
do {                                                                        \
  CUresult err = call;                                                      \
  if (err != CUDA_SUCCESS) {                                                \
    const char* err_str = nullptr;                                          \
    cuGetErrorString(err, &err_str);                                        \
    std::string msg = std::string("CUDA Driver API Error in ") + __FILE__ + \
      ":" + std::to_string(__LINE__) + " : " + (err_str ? err_str : "Unknown"); \
    throw std::runtime_error(msg);                                          \
  }                                                                         \
} while (0)

// Launch a PTX kernel using PyTorch's current context + stream.
// 'tensors' are passed as device pointers. 'scalar_ints' are optional .u32/.u64 params.
void launch_ptx(const std::string& ptx_code,
                const std::string& kernel_name,
                int grid_dim_x, int grid_dim_y, int grid_dim_z,
                int block_dim_x, int block_dim_y, int block_dim_z,
                const std::vector<at::Tensor>& tensors,
                const std::vector<int64_t>& scalar_ints /* optional */) {
  // Basic checks
  for (const auto& t : tensors) {
    TORCH_CHECK(t.is_cuda(), "All tensors must be CUDA tensors");
    TORCH_CHECK(t.is_contiguous(), "All tensors must be contiguous");
  }

  // *** Use current context, do NOT create/destroy your own.
  CUcontext ctx = nullptr;
  CUDA_CHECK(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    // Fall back to primary context of the device of the first tensor
    int dev_index = tensors.empty() ? at::cuda::current_device() : tensors[0].get_device();
    CUdevice cuDevice;
    CUDA_CHECK(cuDeviceGet(&cuDevice, dev_index));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, cuDevice));
    CUDA_CHECK(cuCtxSetCurrent(ctx));
  }

  // Load module
  CUmodule cuModule;
  CUDA_CHECK(cuModuleLoadData(&cuModule, ptx_code.c_str()));

  // Get function
  CUfunction cuFunction;
  CUDA_CHECK(cuModuleGetFunction(&cuFunction, cuModule, kernel_name.c_str()));

  // *** Build argument storage and pointers
  std::vector<CUdeviceptr> dev_ptr_storage;
  dev_ptr_storage.reserve(tensors.size());
  for (const auto& t : tensors) {
    dev_ptr_storage.push_back(reinterpret_cast<CUdeviceptr>(t.data_ptr()));
  }

  // Scalars (keep storage alive)
  std::vector<uint64_t> scalar_storage; // store as 64-bit; PTX .u32 will read lower 32 bits
  scalar_storage.reserve(scalar_ints.size());
  for (auto v : scalar_ints) {
    scalar_storage.push_back(static_cast<uint64_t>(v));
  }

  std::vector<void*> kernel_args;
  kernel_args.reserve(dev_ptr_storage.size() + scalar_storage.size());
  for (auto& d : dev_ptr_storage) kernel_args.push_back(&d);
  for (auto& s : scalar_storage)  kernel_args.push_back(&s);

  // *** Use PyTorch's current stream
  auto torch_stream = at::cuda::getCurrentCUDAStream(); // CUDAStream wrapper
  cudaStream_t rt_stream = torch_stream.stream();       // underlying runtime stream
  
  // Use it with the Driver API
  CUstream cuStream = reinterpret_cast<CUstream>(rt_stream);

  // Sanity: thread/block limits (A10: x<=1024, y<=1024, z<=64; total threads per block<=1024)
  TORCH_CHECK(block_dim_x > 0 && block_dim_y > 0 && block_dim_z > 0, "Invalid block dims");
  TORCH_CHECK((long long)block_dim_x * block_dim_y * block_dim_z <= 1024,
              "Threads per block exceed 1024");

  CUDA_CHECK(cuLaunchKernel(cuFunction,
                            grid_dim_x, grid_dim_y, grid_dim_z,
                            block_dim_x, block_dim_y, block_dim_z,
                            0 /*sharedMemBytes*/,
                            cuStream,
                            kernel_args.data(),
                            nullptr));

  // sync only if you want blocking semantics here; PyTorch will often sync elsewhere
  CUDA_CHECK(cuStreamSynchronize(cuStream));

  // Unload module (keeps context intact)
  CUDA_CHECK(cuModuleUnload(cuModule));
}

PYBIND11_MODULE(ptx_launcher_module, m) {
  m.def("launch",
        [](const std::string& ptx_code,
           const std::string& kernel_name,
           int gx,int gy,int gz,int bx,int by,int bz,
           const std::vector<at::Tensor>& tensors,
           const std::vector<int64_t>& scalar_ints) {
             launch_ptx(ptx_code, kernel_name, gx,gy,gz, bx,by,bz, tensors, scalar_ints);
           },
        "Launch a PTX kernel");
}
