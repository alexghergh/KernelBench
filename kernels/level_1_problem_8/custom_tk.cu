#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WORKERS (1)
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

// Tile dimensions
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// Global memory descriptors
using a_gl = gl<bf16, -1, -1, -1, -1, st<bf16, TILE_M, TILE_K>>;
using b_gl = gl<bf16, -1, -1, -1, -1, st<bf16, TILE_K, TILE_N>>;
using c_gl = gl<float, -1, -1, -1, -1, st<float, TILE_M, TILE_N>>;

struct micro_globals {
    a_gl a;
    b_gl b;
    c_gl c;
    int K; // Total K dimension to loop over
    
    dim3 grid() { 
        return dim3((c.cols + TILE_N - 1) / TILE_N, 
                   (c.rows + TILE_M - 1) / TILE_M); 
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 224000; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Allocate shared memory tiles
    st<bf16, TILE_M, TILE_K> (&a_s) = al.allocate<st<bf16, TILE_M, TILE_K>>();
    st<bf16, TILE_K, TILE_N> (&b_s) = al.allocate<st<bf16, TILE_K, TILE_N>>();
    st<float, TILE_M, TILE_N> (&c_s) = al.allocate<st<float, TILE_M, TILE_N>>();

    // Register tiles
    rt<bf16, TILE_M, TILE_K, ducks::rt_layout::row> a_reg;
    rt<bf16, TILE_K, TILE_N, ducks::rt_layout::col> b_reg;
    rt<float, TILE_M, TILE_N, ducks::rt_layout::row> accum;
    zero(accum);

    // Loop over K dimension in tiles
    for (int k = 0; k < g.K; k += TILE_K) {
        // if (blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("k: %d\n", k);
        // }

        // Load tiles from global memory
        load(a_s, g.a, {0, k});
        load(b_s, g.b, {0, k});
        __syncthreads();

        // Load to registers
        load(a_reg, a_s);
        load(b_reg, b_s);
        __syncthreads();

        // Perform matrix multiplication on tiles
        mma_AB(accum, a_reg, b_reg, accum);

        // if (blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("accum: %f\n", accum);
        // }

        __syncthreads();
    }

    // Store result
    store(c_s, accum);
    __syncthreads();
    store(g.c, c_s, {0, 0});
    __syncthreads(); // SIMON: I added this

}

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

PYBIND11_MODULE(tk_kernels, m) {
    m.doc() = "tk_kernels python module";
    BIND_KERNEL(m, "micro_tk", micro_tk, micro_globals, a, b, c, K);
    BIND_FUNCTION(m, "dispatch_micro", dispatch_micro, micro_globals, a, b, c, K);
}