# Reproducing Sakana's Result

We focus on invesigating and understanding the results of Sakana's kernels.

We have thoroughly examined 2 problems. There might be more, and we will continue to update.
* Level 1 Problem 15: `15_Matmul_for_lower_triangular_matrices`
* Level 2 Problem 23: `23_Conv3d_GroupNorm_Mean`

For each problem, we put the kernel code in a folder with the following structure:
* We have the original code from Sakana, which is `_sakana.cu`. This is pure CUDA code and then bind to the model `forward` function using `pybind11`.
* We have the code in the KernelBench format `ModelNew`, which is `_kernelbench.py`. This is a PyTorch module with custom inline CUDA kernel, which is the KernelBench task format.

### Note on Sakana's Eval System

Describe Sakana's eval, describe Kernel Bench's eval.
See example of how Sakana evaluate thier kernel, provided by [Sakana paper author](https://x.com/RobertTLange/status/1892489402070220989). 

You can use `scripts/run_and_check.py` to evaluate **using the KernelBench Eval code**. 

### Level 1 Problem 15: `15_Matmul_for_lower_triangular_matrices`

To use the KernelBench eval on this problem, you can run the following command:
```
python3 scripts/run_and_check.py ref_origin=kernelbench level=1 problem_id=15 kernel_src_path=sakana_kernels/level1_problem15/15_Matmul_for_lower_triangular_matrices_kernelbench.py
```


For this problem, the CUDA kernel is initialized with a 1D grid as follows:
```
const int threadsPerBlock = 256; // Increased thread count per block
const int numBlocks = N;
triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
    A.data_ptr<float>(),
    B.data_ptr<float>(),
    C.data_ptr<float>(),
    N
);
```
However, in the actual kernel, we compute the row and column a thread computes as if we're using a 2D grid & block:
```
const int row = blockIdx.y * blockDim.y + threadIdx.y;
const int col = blockIdx.x * blockDim.x + threadIdx.x;
```
In this case: `blockIdx.y` will always be 0, `blockDim.y` will always be 1, and `threadIdx.y` will always be 0. So, the value of `row` will always be 0. This is reflected when we look at the result matrices computed by the kernel:

TODO: output of sakana vs output of kernelbench vs output of reference, show that only the first row is computed correctly and everything else is incorrect.
ASIDE: verify whether the entire first row is computed correctly vs just the first element
    row is always 0, and in the kernel we check if (col <= row) so we should in theory only set `C[0][0]` and everything else should be 0.

We can fix this one of two ways:
1. Configure a 2D grid/block:
```
dim3 block(16, 16);
dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
triangular_mm_kernel<<<grid, block>>>(
    A.data_ptr<float>(),
    B.data_ptr<float>(),
    C.data_ptr<float>(),
    N
)
```
2. Determine row/col by correctly indexing in 1D:
```
const int threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
const int row = threadIdx / N;
const int col = threadIdx % N;
if (row < N && col < N) {
    if (col <= row) {
        // Lower triangle computation
        float sum = 0.0f;
        // Process elements in chunks to improve cache utilization
        # pragma unroll 8
        for (int k = col; k <= row; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    } else {
        // Upper triangle (set to zero)
        C[row * N + col] = 0.0f;
    }
}
```

### Level 2 Problem 23: `23_Conv3d_GroupNorm_Mean`

TODO: the problem is it just returned 0s.....


To use the KernelBench eval on this problem, you can run the following command:
```
python3 scripts/run_and_check.py ref_origin=kernelbench level=2 problem_id=23 kernel_src_path=sakana_kernels/level2_problem23/23_Conv3d_GroupNorm_Mean_kernelbench.py
```




On NVIDIA L40S, we see with our eval code.
```
========================================
[Eval] Kernel eval result: compiled=True correctness=True metadata={'hardware': 'NVIDIA L40S', 'device': 'cuda:0', 'correctness_trials': '(5 / 5)'} runtime=0.0327 runtime_stats={'mean': 0.0327, 'std': 0.00188, 'min': 0.0307, 'max': 0.0481, 'num_trials': 100, 'hardware': 'NVIDIA L40S', 'device': 'cuda:0'}
----------------------------------------
[Timing] PyTorch Reference Eager exec time: 1.26 
[Timing] PyTorch Reference torch.compile time: 0.704 ms
[Timing] Custom Kernel exec time: 0.0327 ms
----------------------------------------
[Speedup] Speedup over eager: 38.53x
[Speedup] Speedup over torch.compile: 21.53x
========================================
```


### Takeaways
We appreciate Sakana's effort in providing the kernel code and the evaluation system. This level of transparency help the community understand and reproduce, enabling future progress in this direction.

We will continue working on making the eval robust.









