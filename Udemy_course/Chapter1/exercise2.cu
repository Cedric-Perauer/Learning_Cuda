#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void mem_test(int *input)
{
 int tid = blockIdx.x + blockIdx.y * gridDim.x
 + gridDim.x * gridDim.y * blockIdx.z;
int gid = tid * (blockDim.x * blockDim.y * blockDim.z)
 + (threadIdx.z * (blockDim.x * blockDim.y))
 + (threadIdx.y * blockDim.x) + threadIdx.x;
 printf("tid is : %d, gid is : %d, input is : %d \n",tid,gid,input[gid]);   

}

int main()
{

int size = 64; 
int byte_size = size * sizeof(int); 
int * h_input;
h_input = (int *)malloc(byte_size); 
time_t t; 
srand((unsigned)time(&t)); 
for(int i=0; i< size; ++i)
{
    h_input[i] = (int)(rand() & 0xff); 
}
int *d_input; 
cudaMalloc((void **)&d_input,byte_size); 


dim3 block(2,2,2); 
dim3 grid(2,2,2); 

mem_test<<<grid,block>>> (d_input); 
cudaDeviceSynchronize(); 
cudaDeviceReset(); 
return 0;
}
