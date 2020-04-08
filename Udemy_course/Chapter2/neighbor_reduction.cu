#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include "timer.h"
#include "utils.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//reduction neighbored pairs kernel
__global__ void redunction_neighbored_pairs(int * input,  //has warp divergence !!!!!!!!!!!!!!!
	int * temp, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size)
		return;

	for (int offset = 1; offset <= blockDim.x/2; offset *= 2)
	{
		if (tid % (2 * offset) == 0)
		{
			input[gid] += input[gid + offset];
		}

		__syncthreads();
	}

	if (tid == 0) //only thread 0 can write 
	{
		temp[blockIdx.x] = input[gid];
	}
}


//reduction neighbored pairs kernel
//Belloch 
__global__ void redunction_neighbored_pairs_improved(int * input,  //has warp divergence only in last iteration
	int * temp, int size) 
{
	int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    
    //local data block 
    int *i_data = input + blockDim.x * blockIdx.x; 
	if (gid > size)
		return;

	for (int offset = 1; offset <= blockDim.x/2; offset *= 2)
	{
        int index = 2 * offset* tid; 
        if(index < blockDim.x)
        {
            i_data[index] += i_data[index + offset]; 
        } 

		__syncthreads();
	}

	if (tid == 0) //only thread 0 can write 
	{
		temp[blockIdx.x] = input[gid];
	}
}


__global__ void interleaved_pairs_unroll_loop(int * input,  //has warp divergence only in last iteration
	int * temp, int size) 
{
	int tid = threadIdx.x;

    int offset = blockDim.x * blockIdx.x *2; 
    int index = offset + tid; 
    int *i_data = input + offset; 
    if((index + blockDim.x)<size)
    {
        input[index] += input[index + blockDim.x]; 
    }
    __syncthreads(); 
	
	for (int offset = blockDim.x/2; offset >0; offset /= 2)
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

__global__ void reduction_interleaved_pairs(int * input,  //unroll two thread blocks into one 
	int * temp, int size) 
{
	int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    
    //local data block 
	if (gid > size)
		return;

	for (int offset = blockDim.x/2; offset >0; offset /= 2)
	{
        if(tid < offset)
        {
            input[gid] += input[gid + offset]; 
        } 

		__syncthreads();
	}

	if (tid == 0) //only thread 0 can write 
	{
		temp[blockIdx.x] = input[gid];
    }
}

int main(int argc, char ** argv)
{
	printf("Running neighbored pairs reduction kernel \n");

	int size = 1 << 27; //128 Mb of data
	int byte_size = size * sizeof(int);
	int block_size = 128;

	int * h_input, *h_ref,*h_ref2,*h_ref3;
	h_input = (int*)malloc(byte_size);

	initialize(h_input, size, INIT_RANDOM);

	//get the reduction result from cpu
	int cpu_result = reduction_cpu(h_input,size);

	dim3 block(block_size);
	dim3 grid(size/ block.x);

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
    redunction_neighbored_pairs << <grid, block >> > (d_input,d_temp, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock(); 
    printf("Time passed for Hillis-Steel GPU : %f ms\n",(double)(double)(end-start)/(CLOCKS_PER_SEC)*1000); 
    
    start = clock(); 
    redunction_neighbored_pairs_improved<< <grid, block >> > (d_input2,d_temp2, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock(); 
    printf("Time passed for Belloch GPU : %f ms\n",(double)(double)(end-start)/(CLOCKS_PER_SEC)*1000);
    
    start = clock(); 
    reduction_interleaved_pairs<< <grid, block >> > (d_input3,d_temp3, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock(); 
    printf("Time passed for Interleaved Pairs GPU : %f ms\n",(double)(double)(end-start)/(CLOCKS_PER_SEC)*1000);
    

	cudaMemcpy(h_ref,d_temp, temp_array_byte_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref2,d_temp2, temp_array_byte_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref3,d_temp3, temp_array_byte_size,cudaMemcpyDeviceToHost);
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
