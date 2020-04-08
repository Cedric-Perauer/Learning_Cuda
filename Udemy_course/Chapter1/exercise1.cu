#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_details()
{
 printf("blockIdx.x : %d, blockIdx.y : %d, blockIdx.z : %d, threadIdx.x : %d,threadIdx.y : %d, threadIdx.z : %d,gridDim.x : %d, gridDim.y : %d, gridDim.z : %d \n", blockIdx.x, blockIdx.y, blockIdx.z, 
	 threadIdx.x, threadIdx.y, threadIdx.z, gridDim.x, gridDim.y, gridDim.z); 

}

int main(int argc, char** argv)
{
printf("running neighbored pairs reduction\n"); 

int size = 1 << 27; //28MB of Data
int byte_size = size * sizeof(int); 

print_details<<<grid,block>>>(); 

cudaDeviceSynchronize(); 
cudaDeviceReset(); 
return 0;
}
