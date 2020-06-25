/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <climits>

const int BLOCK_SIZE = 1024; 

void min_kernel(float * d_out, float *d_in, int size)
{  

   extern __shared__ float s_data[]; //is located in kernel call -> 3rd element needed with size of bytes
   int Id = threadIdx.x + blockDim.x * blockIdx.x; 
   int tid = threadIdx.x; 
   s_data[tid] = d_in[tid]; //load to shared memory
   __synchthreads(); 
   for(unsigned int s = blockDim.x/2; s > 0; s >>=1)
   {
      if(tid>=size)
      {
         s_data[Id] = s_data[Id+s] < s_data[Id] ? s_data[Id+s] : s_data[Id];  //setmin
      }
      __synchthreads(); 
   }

   if(tid==0)//only thread 0 can write to output array
   {
      d_out[blockIdx.x] = s_data[0]; 
   }

}


__global__ void max_kernel(float * d_out, float *d_in)
{  

   extern __shared__ float s_data[]; //is located in kernel call -> 3rd element needed with size of bytes
   int Id = threadIdx.x + blockDim.x * blockIdx.x; 
   int tid = threadIdx.x; 
   s_data[tid] = d_in[tid]; //load to shared memory
   __synchthreads();  
   for(unsigned int s = blockDim.x/2; s > 0; s >>=1)
   {
      if(tid<s)
      {
         s_data[Id] = s_data[Id+s] > s_data[Id] ? s_data[Id+s] : s_data[Id];  //setmax
      }
      __synchthreads(); 
   }

   if(tid==0)//only thread 0 can write to output array
   {
      d_out[blockIdx.x] = s_data[0]; 
   }
}

__global__ void histo_atomic(unsigned int *out_histo,const float *d_in, int num_bins, int size,float min_val,float range)
{
   int tid = threadIdx.x; 
   int id = tid + blockIdx.x * blockIdx.x; 
   if(tid >= size)
   {
      return; 
   }
   int bin = ((d_in[id]-min_val)*num_bins)/range; 
   bin = bin == num_bins ? num_bins -1 : bin; //max value bin is last bin of the histogram
   atomicAdd(&(out_histo[bin]),1); 
}

__global__ void scan_hillis_steele(unsigned int *d_out,unsigned int *d_in, unsigned int size)
{
   extern __shared__ unsigned int temp[]; 
   int tid = threadIdx.x; 
   int i_0 = 0; 
   int i_1 = 1;
   if(tid>0)  
   {
      temp[tid] = d_in[tid-1]; //exclusive
   } 
   else
   {
      temp[tid] = 0; 
   }

   __synchthreads(); 

   for(int j = 1; j < size; j <<=1)
   {
      i_0 = 1 - i_0; 
      i_1 = 1 - i_1;
      if(tid>=j)
      {
         temp[size*i_0+tid] = temp[size*i_1+tid] + temp[size*i_1+tid-j]; 
      } 
      else 
      {
         temp[size*i_0 + tid] = temp[size*i_0+tid]; 
      }
      __synchthreads(); 

   }
   d_out[tid] = temp[i_0*size+tid]; 
}


float reduce_min(const float* const d_logLuminance, int input_size)
{
   int threads = BLOCK_SIZE; 
   float *d_cur = NULL; 
   int size = input_size; 
   int blocks = ceil(1.0*size/threads);
   while(true)
   {
      float *d_out; //intermediate results
      checkCudaErrors(cudaMalloc(&d_out,blocks*sizeof(float))); 
      if(d_cur==NULL)
      {
         min_kernel<<<blocks,threads,threads*sizeof(float)>>>(d_out,d_logLuminance,size); 
      }
      else
      {
         min_kernel<<<blocks,threads,threads*sizeof(float)>>>(d_out,d_cur,size); 
      }
      cudaDeviceSynchronize(); 
      checkCudaErrors(cudaGetLastError());

      //free last intermediate result
		if (d_current_in != NULL) checkCudaErrors(cudaFree(d_current_in));

      if(blocks==1)
      {
         float h_output; 
         checkCudaErrors(cudaMemcpy(%h_output,d_out,sizeof(float),cudaMemcpyDeviceToHost)); 
         return h_output; 
      }
      size = blocks; 
      blocks = ceil(1.0f*size/threads); 
      if(blocks==0)
         blocks++; 
      d_cur = d_out; 
      
   } 
}

float reduce_max(const float* const d_logLuminance, int input_size)
{
   int threads = BLOCK_SIZE; 
   float *d_cur = NULL; 
   int size = input_size; 
   int blocks = ceil(1.0*size/threads);
   while(true)
   {
      float *d_out; //intermediate results
      checkCudaErrors(cudaMalloc(&d_out,blocks*sizeof(float))); 
      if(d_cur==NULL)
      {
         max_kernel<<<blocks,threads,threads*sizeof(float)>>>(d_out,d_logLuminance,size); 
      }
      else
      {
         max_kernel<<<blocks,threads,threads*sizeof(float)>>>(d_out,d_cur,size); 
      }
      cudaDeviceSynchronize(); 
      checkCudaErrors(cudaGetLastError());

      //free last intermediate result
		if (d_current_in != NULL) checkCudaErrors(cudaFree(d_current_in));

      if(blocks==1)
      {
         float h_output; 
         checkCudaErrors(cudaMemcpy(%h_output,d_out,sizeof(float),cudaMemcpyDeviceToHost)); 
         return h_output; 
      }
      size = blocks; 
      blocks = ceil(1.0f*size/threads); 
      if(blocks==0)
         blocks++; 
      d_cur = d_out; 
      
   } 
}


unsigned int* compute_histogram(const float* const d_logLuminance, int numBins, int input_size, float minVal, float rangeVals) 
{  
   int threads = BLOCK_SIZE;
	unsigned int* d_histo;
	checkCudaErrors(cudaMalloc(&d_histo, numBins * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_histo, 0, numBins * sizeof(unsigned int)));
	int blocks = ceil(1.0f*input_size / threads);
	histo_atomic << <blocks, threads >> >(d_histo, d_logLuminance, numBins, input_size, minVal, rangeVals);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	return d_histo;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{  

   //1) min and max
   int input_size = numRows * numCols; 
   min_logLum = reduce_min(d_logLuminance,input_size);
   max_logLum = reduce_max(d_logLuminance,input_size);
   //2) Range 
   float range = max_logLum - min_logLum; 
   //3) Histogram Step
   unsigned int *d_histo =  compute_histogram(d_logLuminance,input_size,min_logLum,range);
   //4) scan 
   scan_hillis_steele <<<1,numBins,2*numBins*sizeof(unsigned int) >>>(d_cdf,histo,numBins);
   checkCudaErrors(cudaFree(histo));  
   

}
