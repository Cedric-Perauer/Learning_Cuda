#include <stdio.h>
#include "reduction.h"

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/
__global__ void
reduction_kernel_2(float *g_out, float *g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[threadIdx.x] = (idx_x < size) ? g_in[idx_x] : 0.f;

    __syncthreads();

    // do reduction
    // sequential addressing
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0)
        g_out[blockIdx.x] = s_data[0];
}

int reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    int n_blocks = (size + n_threads - 1) / n_threads;
    reduction_kernel_2<<< n_blocks, n_threads, n_threads * sizeof(float), 0 >>>(g_outPtr, g_inPtr, size);
    return n_blocks;
}