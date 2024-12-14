#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WORKERS (1) // Please keep this to 1 for now
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)  // number of threads is num_workers * num_threads_per_worker (32)

// MODEL TODO: SET DIMENSIONS OF INPUTS
#define M 16 // rows per input matrix x
#define N 16 // columns per input matrix x

/* 
MODEL TODO: DEFINE GLOBAL MEMORY DESCRIPTORS
Format: gl<bf16, -1, -1, -1, -1,  st<bf16, M, N>>; 
Make sure you have 4 dimensions (-1, -1, -1, -1) in example above
gl: indicates global layout
bf16: indicates the data type
four dimmensions: {batch, head, depth, width} (-1 is runtime dimension)
st: when loading from global tensor at some {b, h, d, w} index, this is the shape of the tile that will be loaded to shared memory
*/
using x_gl  = gl<bf16, -1, -1, -1, -1,  st<bf16, M, N>>;  // input is bfloat16
using o_gl  = gl<float, -1, -1, -1, -1, st<float, M, M>>; // output is float32

/*
CREATE GLOBAL MEMORY DESCRIPTORS
CREATE GRID AND BLOCK LAUNCH DIMENSIONS
*/
struct micro_globals {
    x_gl x, y;
    o_gl o;
    dim3 grid()  { return dim3(o.depth, o.batch); } // dimensions we parallelize over (e.g., batches, heads)
    dim3 block() { return dim3(NUM_THREADS); } // number of threads per threadblock
    size_t dynamic_shared_memory() { return 224000; } // shared memory size for H100
    
};

/*
ACTUAL CUDA KERNEL
*/
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    // get current position in grid
    int head = blockIdx.x;
    int batch = blockIdx.y;

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // inputs should be in bf16, outputs should be in float
    // tiles can be in row or col layout
    st<bf16, M, N> (&x_s) = al.allocate<st<bf16, M, N>>(); // bf16 tiles
    st<bf16, M, N> (&y_s) = al.allocate<st<bf16, M, N>>(); // bf16 tiles
    st<float, M, M> (&o_s) = al.allocate<st<float, M, M>>(); // float tiles

    // register memory
    // inputs should be in bf16, outputs should be in float
    rt<bf16, M, N, ducks::rt_layout::row> x_reg; // bf16 register
    rt<bf16, M,N, ducks::rt_layout::row> y_reg; // bf16 register
    rt<float, M, M, ducks::rt_layout::row> accum_tile;  // float register
    zero(accum_tile);

    // load from HBM to shared
    load(x_s, g.x, {batch, head, 0, 0});
    load(y_s, g.y, {batch, head, 0, 0});
    __syncthreads();

    // load from shared to register
    load(x_reg, x_s);
    load(y_reg, y_s);
    __syncthreads();

    // now do the matmul and accumulate to accum_tile
    mma_ABt(accum_tile, x_reg, y_reg, accum_tile); // o = torch.matmul(x, x.transpose(1, 2))
    __syncthreads();

    // store from register to shared
    store(o_s, accum_tile);
    __syncthreads();

    // store from shared to HBM
    store(g.o, o_s, {batch, head, 0, 0});
    __syncthreads();
}

/*
DISPATCH FUNCTION 
*/
void dispatch_micro(micro_globals g) {
    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    cudaDeviceSynchronize();
}

/* 
PYTHON BINDINGS
*/
PYBIND11_MODULE(tk_kernels, m) {
    m.doc() = "tk_kernels python module";
    BIND_KERNEL(m, "micro_tk", micro_tk, micro_globals, x, y, o); // For wrapping kernels directly.
    BIND_FUNCTION(m, "dispatch_micro", dispatch_micro, micro_globals, x, y, o); // For host functions that wrap the kernel.
}
