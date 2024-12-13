#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WORKERS (4)
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

// Tile dimensions
#define TILE_M 64  // We'll process 64 rows at a time
#define TILE_N 16  // Full width since it's small

using a_gl = gl<bf16, -1, -1, -1, -1, st<bf16, TILE_M, TILE_N>>;  // Input A is bfloat16
using b_gl = gl<bf16, -1, -1, -1, -1, st<bf16, TILE_N, TILE_M>>;  // Input B is bfloat16
using c_gl = gl<float, -1, -1, -1, -1, st<float, TILE_M, TILE_M>>; // Output is float32

struct tall_matmul_globals {
    a_gl a;
    b_gl b;
    c_gl c;
    dim3 grid() { 
        return dim3((a.depth + TILE_M - 1) / TILE_M); // Number of row tiles
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 224000; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void tall_matmul_tk(const __grid_constant__ tall_matmul_globals g) {
    int tile_idx = blockIdx.x;
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Shared memory tiles
    st<bf16, TILE_M, TILE_N> (&a_s) = al.allocate<st<bf16, TILE_M, TILE_N>>();
    st<bf16, TILE_M, TILE_N> (&b_s) = al.allocate<st<bf16, TILE_M, TILE_N>>();
    st<float, TILE_M, TILE_M> (&c_s) = al.allocate<st<float, TILE_M, TILE_M>>();

    // Register tiles
    rt<bf16, TILE_M, TILE_N, ducks::rt_layout::row> a_reg;
    rt<bf16, TILE_M, TILE_N, ducks::rt_layout::row> b_reg;
    rt<float, TILE_M, TILE_M, ducks::rt_layout::row> c_reg;
    zero(c_reg);

    // Load current tile from global to shared
    if (tile_idx * TILE_M < g.a.depth) {
        load(a_s, g.a, {0, 0, tile_idx * TILE_M});
        load(b_s, g.b, {0, 0, tile_idx * TILE_M});
        __syncthreads();

        // Load from shared to register
        load(a_reg, a_s);
        load(b_reg, b_s);
        __syncthreads();

        // Perform matrix multiplication
        mma_ABt(c_reg, a_reg, b_reg, c_reg);
        __syncthreads();

        // Store result
        store(c_s, c_reg);
        __syncthreads();
        
        store(g.c, c_s, {0, 0, tile_idx * TILE_M});
    }
}

void dispatch_tall_matmul(tall_matmul_globals g) {
    unsigned long mem_size = 50480;
    cudaFuncSetAttribute(
        tall_matmul_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    tall_matmul_tk<<<g.grid(), g.block(), mem_size>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    m.doc() = "tk_kernels python module";
    BIND_KERNEL(m, "tall_matmul_tk", tall_matmul_tk, tall_matmul_globals, a, b, c);
    BIND_FUNCTION(m, "dispatch_tall_matmul", dispatch_tall_matmul, tall_matmul_globals, a, b, c);
}