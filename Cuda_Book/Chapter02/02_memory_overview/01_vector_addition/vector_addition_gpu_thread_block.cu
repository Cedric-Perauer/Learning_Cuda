#include<stdio.h>
#include<stdlib.h>

#define N 512

void host_add(int *a, int *b, int *c) {
	for(int idx=0;idx<N;idx++)
		c[idx] = a[idx] + b[idx];
}

__global__ void device_add(int *a, int *b, int *c) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;
        c[index] = a[index] + b[index];
}


//basically just fills the array with index.
void fill_array(int *data) {
	for(int idx=0;idx<N;idx++)
		data[idx] = idx;
}

void print_output(int *a, int *b, int*c) {
	for(int idx=0;idx<N;idx++)
		printf("\n %d + %d  = %d",  a[idx] , b[idx], c[idx]);
}
int main(void) {
	int *a, *b, *c;
        int *d_a, *d_b, *d_c; // device copies of a, b, c
	int threads_per_block=0, no_of_blocks=0;

	int size = N * sizeof(int);

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); fill_array(a);
	b = (int *)malloc(size); fill_array(b);
	c = (int *)malloc(size);

        // Alloc space for device copies of a, b, c
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);

       // Copy inputs to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	threads_per_block = 4;
	no_of_blocks = N/threads_per_block;	
	device_add<<<no_of_blocks,threads_per_block>>>(d_a,d_b,d_c);

        // Copy result back to host
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	print_output(a,b,c);

	free(a); free(b); free(c);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);



	return 0;
}
