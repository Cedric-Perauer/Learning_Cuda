#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include "utils.h"
#include "timer.h"


void cpu_add(int *a,int*b,int*c,int *cpu_results,int size)
{
    for(int i = 0; i < size; ++i)
    {
        cpu_results[i] = a[i] + b[i] + c[i]; 
    }
}



__global__ void gpu_add(int *a,int*b,int*c,int *gpu_results,int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x; 
    if(gid<size)
    {
        gpu_results[gid] = a[gid] + b[gid] + c[gid]; 
    }
}

void compare_array(int *a,int *b,int size)
{
    for(int i = 0; i < size;++i)
    {
        if(a[i]!=b[i])

        {    fprintf(stderr,"Arrays are different"); }
    }
}

int main()
{

int size = 4194304; 
int byte_size = size * sizeof(int); 

int  *cpu_results = (int*)malloc(byte_size);
int *gpu_results = (int*)malloc(byte_size); 
clock_t cpum_start, cpum_end; 
cpum_start = clock(); 
int * a;
a = (int *)malloc(byte_size); 
int *b;
b = (int *)malloc(byte_size); 
int * c;
c = (int *)malloc(byte_size); 
cpum_end = clock(); 
printf("Cpu Memory allocatio took : %f \n",(double)(double)(cpum_end-cpum_start)/(CLOCKS_PER_SEC)); 
time_t t; 
srand((unsigned)time(&t)); 
for(int i=0; i< size; ++i)
{
    a[i] = (int)(rand() & 0xff); 
}

for(int i=0; i< size; ++i)
{
    b[i] = (int)(rand() & 0xff); 
}

for(int i=0; i< size; ++i)
{
    c[i] = (int)(rand() & 0xff); 
}

memset(gpu_results,0,byte_size); 
memset(cpu_results,0,byte_size); 

cpum_start = clock(); 
cpu_add(a,b,c,cpu_results,size);
cpum_end = clock(); 
printf("CPU execution took : %f \n",(double)(double)(cpum_end-cpum_start)/(CLOCKS_PER_SEC)); 

int *d_output,*d_a,*d_b,*d_c; 
cpum_start = clock(); 
checkCudaErrors(cudaMalloc((void **)&d_output,byte_size)); 
checkCudaErrors(cudaMalloc((void **)&d_a,byte_size)); 
checkCudaErrors(cudaMalloc((void **)&d_b,byte_size)); 
checkCudaErrors(cudaMalloc((void **)&d_c,byte_size)); 
cpum_end = clock(); 
printf("GPU Memory allocation took : %f \n",(double)(double)(cpum_end-cpum_start)/(CLOCKS_PER_SEC)); 
cpum_start = clock(); 
checkCudaErrors(cudaMemcpy(d_a,a,byte_size,cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_b,b,byte_size,cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_c,c,byte_size,cudaMemcpyHostToDevice));
cpum_end = clock(); 
printf("CPU-GPU Memory copy took : %f \n",(double)(double)(cpum_end-cpum_start)/(CLOCKS_PER_SEC));


const int block_size[] = {64,128,256,512}; 
for(int i =0 ; i < sizeof(block_size)/sizeof(block_size[0]);++i)
{
    dim3 block(block_size[i]); 
    dim3 grid(size/block_size[i] +1);

    cpum_start = clock(); 
    gpu_add<<<grid,block>>> (d_a,d_b,d_c,d_output,size); 
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
    cpum_end = clock(); 
    printf("GPU execution took with block size %d : %f \n",block_size[i],(double)(double)(cpum_end-cpum_start)/(CLOCKS_PER_SEC)); 
}



//result back to host
checkCudaErrors(cudaMemcpy(gpu_results,d_output,byte_size,cudaMemcpyDeviceToHost));
compare_array(cpu_results,gpu_results,size); 
checkCudaErrors(cudaFree(d_a));
checkCudaErrors(cudaFree(d_b));
checkCudaErrors(cudaFree(d_c));
checkCudaErrors(cudaFree(d_output));

free(gpu_results);
free(a);
free(b);
free(c);
free(cpu_results);


return 0;
}
