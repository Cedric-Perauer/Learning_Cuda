#include<stdio.h> 
#include<stdlib.h> 


//cuda 
#include <cuda_runtime.h> 
#include <helper_timer.h>
#include "reduction.h"  



void run_benchmark(void (*reduce)(float *, float *, int, int),
                   float *d_outPtr, float *d_inPtr, int size)
{
    int num_threads = 256;
    int test_iter = 100;

    // warm-up
    reduce(d_outPtr, d_inPtr, num_threads, size);

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    ////////
    // Operation body
    ////////
    for (int i = 0; i < test_iter; i++)
    {
        cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice);
        reduce(d_outPtr, d_outPtr, num_threads, size);
        cudaDeviceSynchronize();
    }

    // getting elapsed time
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    // Compute and print the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer) / (float)test_iter;
    float bandwidth = size * sizeof(float) / elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);
}

void init_input(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        // Keep the numbers small so we don't get truncation error in the sum
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

float get_cpu_result(float *data, int size)
{
    double result = 0.f;
    for (int i = 0; i < size; i++)
        result += data[i];

    return (float)result;
}

int main(int argc,char * argv[]){ 
float *h_in; 
float *d_in, *d_out; 

unsigned int size 1 << 24; //bit shift by 24 

float result_h, result_d; 

srand(2019); 

//Memory Allocation CPU 
h_in = (float *)malloc(size * sizeof(float)); 

//data init with random values 
init_input(h_in,size);

//GPU allocation 
CudaMalloc((void **)&d_in,size * sizeof(float)); 
CudaMalloc((void **)&d_out,size * sizeof(float)); 

cudaMemcpy(d_in,h_in,size * sizeof(float), cudaMemcpyHostToDevice); 

//Kernel Run 
run_benchmark(global_reduction, d_out, d_in,size); 
cudaMemcpy(&result_h, &d_out[0],size * sizeof(float), cudaMemcpyDeviceToHost); 



} 

	





