#include <stdio.h> 
#include <stdlib.h> 

__global__ void global_reduction_kernel(float *data_out, float *data_in, int stride, int size) { 
	int idx_x = threadIdx.x + blockIdx.x * blockDim.x; 

	if(idx_x + stride < size) { 
	data_out[idx_x] += data_in[idx_x + stride]; //same at every step only stride differs 
	} 
} 


void global_reduction(float *d_out, float *d_in, int n_threads, int size) { 
int blocks_n = (size + n_threads - 1) /n_threads; 

for(int stride = 1; stride < size; stride *= 2) { 
	global_reduction_kernel<<<blocks_n,n_threads>>>(d_out,d_in,stride,size); 
} 
}  
