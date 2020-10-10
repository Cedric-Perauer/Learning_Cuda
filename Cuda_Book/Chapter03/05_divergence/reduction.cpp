#include <stdio.h> 
#include <stdlib.h> 

//cuda includes 
#include <cuda_runtime.h> 
#include <helper_timer.h> 

#include  "reduction.h" 

void run_benchmark(int (*reduce)(float*, float*, int, int),
                   float *d_outPtr, float *d_inPtr, int size);
void init_input(float* data, int size);
float get_cpu_result(float *data, int size);
void message()
{
    puts("Invalid reduction request!! 0-1 are avaiable.");
    exit(EXIT_FAILURE);
}

//Main Part 
int main(int argc, char *argv[]) 
{
	float *h_in,*d_in, *d_out; 

	unsigned int size = 1 << 24; 

	float result_host, gpu_res; 
	int mode = -1; 
	srand(2019); 
	//allocate memory 
	h_in = (float*)malloc(size * sizeof(float));

        init_input(h_in,size); 
	result_host = get_cpu_result(h_in,size); 
	//GPU 
	cudaMalloc((void**)&d_in, size *sizeof(float)); 	
	cudaMalloc((void**)&d_out, size *sizeof(float)); 
	cudaMemcpy(d_in,h_in,size*sizeof(float),cudaMemcpyHostToDevice);
        
	//get result by running kernel 
	run_benchmark(reduction,d_out,d_in,size); 
	cudaMempy(&gpu_res,&d_out[0],sizeof(float),cudaMemcpyDeviceToHost);
        printf("host: %f, device %f\n", result_host, gpu_res);

	//Terminates Memory 
	cudaFree(d_out); 
	cudaFree(d_in); 
	free(h_in); 

	return 0; 
}




void
run_reduction(int (*reduce)(float*, float*, int, int), 
              float *d_outPtr, float *d_inPtr, int size, int n_threads)
{
    //printf("size: %d\n", size);
    while(size > 1) {
        size = reduce(d_outPtr, d_inPtr, size, n_threads);
        //printf("size: %d\n", size);
    }
}

void
run_benchmark(int (*reduce)(float*, float*, int, int), 
              float *d_outPtr, float *d_inPtr, int size)
{
    int num_threads = 256;
    int test_iter = 5;

    // warm-up
    reduce(d_outPtr, d_inPtr, size, num_threads);

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    ////////
    // Operation body
    ////////
    for (int i = 0; i < test_iter; i++) {
        cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice);
        run_reduction(reduce, d_outPtr, d_outPtr, size, num_threads);
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

void
init_input(float *data, int size)
{
    for (int i = 0; i < size; i++) {
        // Keep the numbers small so we don't get truncation error in the sum
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

float
get_cpu_result(float *data, int size)
{
    double result = 0.f;
    for (int i = 0; i < size; i++)
        result += data[i];

    return (float)result;
}
	
