#include<stdio.h>
#include<stdlib.h>

#define N 2048
#define BLOCK_SIZE 32 


//naive way of implementing it, uncoalecesd memory access
__global__ void matrix_transpose_naive(int *in, int *out) {
	int index_x = threadIdx.x + blockDim.x * blockIdx.x; 
	int index_y = threadIdx.y + blockDim.y * blockIdx.y; 

	int idx = index_x + index_y * N;
	int idx_transpose = index_x * N + index_y; //just swith x and y indeces
	out[idx] = in[idx_transpose]; 
}

//shared memory, about 3x faster
__global__ void matrix_transpose_shared(int *in, int *out) {
	
	__shared__ int shared_mem[BLOCK_SIZE][BLOCK_SIZE+1]; 
	
	//global indeces
	int idx_x = threadIdx.x + blockIdx.x * blockDim.x; 
	int idx_y = threadIdx.y + blockIdx.y * blockDim.y; 
	
	//local indeces 
	int local_x = threadIdx.x; 
	int local_y = threadIdx.y; 
	int idx = idx_x + idx_y * N; 
	int idx_transpose = idx_x * N + idx_y; 

	//write input into shared memory, coalesced access 
	shared_mem[local_x][local_y] = in[idx]; 
	__syncthreads(); 

	//copy over into global mem for the output
	out[idx_transpose] = shared_mem[local_y][local_x]; 

}
//basically just fills the array with index.
void fill_array(int *data) {
	for(int idx=0;idx<(N*N);idx++)
		data[idx] = idx;
}

void print_output(int *a, int *b) {
	printf("\n Original Matrix::\n");
	for(int idx=0;idx<(N*N);idx++) {
		if(idx%N == 0)
			printf("\n");
		printf(" %d ",  a[idx]);
	}
	printf("\n Transposed Matrix::\n");
	for(int idx=0;idx<(N*N);idx++) {
		if(idx%N == 0)
			printf("\n");
		printf(" %d ",  b[idx]);
	}
}
int main(void) {
	int *a; 
	int *b; 

	int *d_a;
        int *d_b; 
	
	int size = N * N * sizeof(int); 
	
	//host arrays
	a = (int *) malloc(size); 
	fill_array(a); 
	b = (int *) malloc(size); 	
	
	//device array allocation 
	cudaMalloc((void **)&d_a,size); 	
	cudaMalloc((void **)&d_b,size);

	//copy inputs to device
	cudaMemcpy((void **)&d_a,a,size,cudaMemcpyHostToDevice);	
	cudaMemcpy((void **)&d_b,b,size,cudaMemcpyHostToDevice);	

	//declare sizes
	dim3 blocksize(BLOCK_SIZE,BLOCK_SIZE,1); 
	dim3 gridsize(N/BLOCK_SIZE,N/BLOCK_SIZE,1); 

	//launch the kernel 
	matrix_transpose_shared<<<gridsize,blocksize>>>(d_a,d_b);

	//copy results back to host 
	cudaMemcpy(b,d_b,size,cudaMemcpyDeviceToHost); 		
	
	free(a); free(b); 
	cudaFree(d_a); cudaFree(d_b); 	

	return 0;
}
