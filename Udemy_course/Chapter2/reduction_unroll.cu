#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include "timer.h"
#include "utils.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata,  //GPU launches threads, slower due to larger num of kernel launches 
	unsigned int isize)
{
	int tid = threadIdx.x;

	int *idata = g_idata + blockIdx.x*blockDim.x;
	int *odata = &g_odata[blockIdx.x];

	// stop condition   
	if (isize == 2 && tid == 0)
	{
		g_odata[blockIdx.x] = idata[0] + idata[1];
		return;
	}

	// nested invocation   
	int istride = isize >> 1;
	if (istride > 1 && tid < istride)
	{
		// in place reduction    
		idata[tid] += idata[tid + istride];
	}

	// sync at block level   
	__syncthreads();

	// nested invocation to generate child grids 
	if (tid == 0)
	{
		gpuRecursiveReduce << <1, istride >> > (idata, odata, istride);
		cudaDeviceSynchronize();
	}

	// sync at block level again 
	__syncthreads();
}


__global__ void interleaved_pairs_unroll_loop(int * input,  //has warp divergence only in last iteration
	int * temp, int size) 
{
	int tid = threadIdx.x;

    int offset = blockDim.x * blockIdx.x *2; 
    int index = offset + tid; 
    int *i_data = input + offset; //global memory
    if((index + blockDim.x)<size)
    {
        input[index] += input[index + blockDim.x]; 
    }
    __syncthreads(); 
	
	for (int offset = blockDim.x/2; offset >= 0; offset /= 2)
	{
        if(tid < offset)
        {
            i_data[tid] += i_data[tid + offset]; 
        } 

		__syncthreads();
	}

	if (tid == 0) //only thread 0 can write 
	{
		temp[blockIdx.x] = i_data[0]; 
    }
}


__global__ void interleaved_pairs_unroll_warp(int * input,  
	int * temp, int size) 
{
	int tid = threadIdx.x;

    int offset = blockDim.x * blockIdx.x; 
    int *i_data = input + offset; 
    
	
	for (int offset = blockDim.x/2; offset >= 64; offset /= 2)
	{
        if(tid < offset)
        {
            i_data[tid] += i_data[tid + offset]; 
        } 

		__syncthreads();
    }
    
    if(tid<32)
    {
        volatile int *vsmem = i_data; //no cache is used
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

	if (tid == 0) //only thread 0 can write 
	{
		temp[blockIdx.x] = i_data[0]; 
    }
}

__global__ void complete_unroll(int * input,  //has warp divergence only in last iteration
	int * temp, int size) 
{
	int tid = threadIdx.x;

    int offset = blockDim.x * blockIdx.x; 
    int *i_data = input + offset; 
    if(blockDim.x == 1024 && tid <512)
    {
        i_data[tid] += i_data[tid + 512]; 
    }
    __syncthreads(); 
	
	if(blockDim.x == 512 && tid <256)
    {
        i_data[tid] += i_data[tid + 256]; 
    }
    __syncthreads();
    if(blockDim.x == 256 && tid <128)
    {
        i_data[tid] += i_data[tid + 128]; 
    }
    __syncthreads();
    if(blockDim.x == 128 && tid <64)
    {
        i_data[tid] += i_data[tid + 64]; 
    }
    __syncthreads();
    if(tid<32)
    {
        volatile int *vsmem = i_data; //no cache is used
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

	if (tid == 0) //only thread 0 can write 
	{
		temp[blockIdx.x] = i_data[0]; 
    }
}


int main(int argc, char ** argv)
{
	printf("Running neighbored pairs reduction kernel \n");

	int size = 1 << 22; 
	int byte_size = size * sizeof(int);
	int block_size = 128;

	int * h_input, *h_ref,*h_ref2,*h_ref3;
	h_input = (int*)malloc(byte_size);

	initialize(h_input, size, INIT_RANDOM);

	//get the reduction result from cpu
	int cpu_result = reduction_cpu(h_input,size);

	dim3 block(block_size);
    dim3 grid(size/block_size);
	//dim3 grid((size/block_size)+2); //interleaved_pairs_unroll_loop
    

	printf("Kernel launch parameters | grid.x : %d, block.x : %d \n",
		grid.x, block.x);

	int temp_array_byte_size = sizeof(int)* grid.x;
    h_ref = (int*)malloc(temp_array_byte_size);
    h_ref2 = (int*)malloc(temp_array_byte_size);
    h_ref3 = (int*)malloc(temp_array_byte_size);


    int * d_input, *d_temp;
    int * d_input2, *d_temp2;
    int * d_input3, *d_temp3;


	checkCudaErrors(cudaMalloc((void**)&d_input,byte_size));
    checkCudaErrors(cudaMalloc((void**)&d_temp, temp_array_byte_size));
    checkCudaErrors(cudaMalloc((void**)&d_input2,byte_size));
    checkCudaErrors(cudaMalloc((void**)&d_temp2, temp_array_byte_size)); 
    checkCudaErrors(cudaMalloc((void**)&d_input3,byte_size));
    checkCudaErrors(cudaMalloc((void**)&d_temp3, temp_array_byte_size)); 
    

 

	checkCudaErrors(cudaMemset(d_temp, 0 , temp_array_byte_size));
	checkCudaErrors(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));    
    checkCudaErrors(cudaMemset(d_temp2, 0 , temp_array_byte_size));
    checkCudaErrors(cudaMemcpy(d_input2, h_input, byte_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_temp3, 0 , temp_array_byte_size));
    checkCudaErrors(cudaMemcpy(d_input3, h_input, byte_size, cudaMemcpyHostToDevice));


   
      

    clock_t start,end; 
    start = clock(); 
    //interleaved_pairs_unroll_loop << <grid, block >> > (d_input,d_temp, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock(); 
    printf("Time passed for interleaved_pairs_unroll_loop GPU : %f ms\n",(double)(double)(end-start)/(CLOCKS_PER_SEC)*1000); 
    
    start = clock(); 
    //interleaved_pairs_unroll_warp<< <grid, block >> > (d_input2,d_temp2, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock(); 
    printf("Time passed for interleaved_pairs_unroll_warp GPU : %f ms\n",(double)(double)(end-start)/(CLOCKS_PER_SEC)*1000);
    
    start = clock(); 
    complete_unroll<< <grid, block >> > (d_input3,d_temp3, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock(); 
    printf("Time passed for Interleaved Pairs GPU : %f ms\n",(double)(double)(end-start)/(CLOCKS_PER_SEC)*1000);
    

	cudaMemcpy(h_ref,d_temp, temp_array_byte_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref2,d_temp2, temp_array_byte_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref3,d_temp3, temp_array_byte_size,cudaMemcpyDeviceToHost);


    
    int gpu_result = 0;
    int gpu_result2 = 0;
    int gpu_result3 = 0;


	for (int i = 0; i < grid.x; i++)
	{
        gpu_result += h_ref[i];
        gpu_result2 += h_ref2[i];
        gpu_result3 += h_ref3[i];

	}

	//validity check
    compare_results(gpu_result, cpu_result);
    compare_results(gpu_result2, cpu_result);
    compare_results(gpu_result3, cpu_result);


	checkCudaErrors(cudaFree(d_temp));
	checkCudaErrors(cudaFree(d_input));

	free(h_ref);
	free(h_input);

	checkCudaErrors(cudaDeviceReset());
	return 0;
}
