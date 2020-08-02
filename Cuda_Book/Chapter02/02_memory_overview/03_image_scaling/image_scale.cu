#include<stdio.h>
#include"scrImagePgmPpmPackage.h"


const float SCALE_RATIO = 0.5; 


//Step 1: Texture Memory 
texture<unsigned char, 2, cudaReadModeElementType> text;

//Kernel to calculate resized size 

__global__ void resized(unsigned char *imgData, int width, float scale_factor, cudaTextureObject_t texObj) {
    const unsigned  int tidX = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned  int tidY = blockIdx.y * blockDim.y + threadIdx.y; 

    const unsigned idx = tidY * width + tidX; 

    //Read texture mem to CUDA Kernel 

    imgData[idx] = tex2D<unsigned char>(texObj,(float)(tidX*scale_factor),(float)(tidY*scale_factor)); 

}


int main(int argc, char *argv[])
{
    int h = 0; 
    int w = 0; 
    int scaled_h = 0; 
    int scaled_w = 0; 
    
    char inputStr[1024] = {"aerosmith-double.pgm"};
	char outputStr[1024] = {"aerosmith-double-scaled.pgm"};
    float ratio = SCALE_RATIO; 
    unsigned char *data; 
    unsigned char *scaled_data; 
    unsigned char *dscaled_data; //for GPU 
    
    cudaError_t returnValue; 

    //channel description to link with texture 
    cudaArray* cu_array; 
    cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned; 
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8,0,0,0,kind); 
    

    get_PgmPpmParams(inputStr,&h,&w); //function to get width and height of image; 
    data = (unsigned char*)malloc(h*w*sizeof(unsigned char)); 
    printf("\n Reading image width height and width [%d][%d]", h, w); 
    scr_read_pgm(inputStr,data,h,w); //load an image       
    
    
    scaled_h = (int)(h *ratio); 
    scaled_w = (int)(w *ratio); 
    scaled_data = (unsigned char*) malloc(scaled_h*scaled_w*sizeof(unsigned char)); 
    printf("\n scaled image width height and width [%d][%d]", scaled_h, scaled_w);

    //CUDA MALLOC 
    returnValue = cudaMallocArray(&cu_array,&channelDesc,w,h);
    returnValue = (cudaError_t)(returnValue | cudaMemcpyToArray(cu_array,0,0,data,h*w*sizeof(unsigned char),cudaMemcpyHostToDevice));
    if(returnValue != cudaSuccess) printf("\n Got error while running CUDA API Array Copy");

    //texture specify 
    struct cudaResourceDesc resDesc; 
    memset(&resDesc,0,sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray; 
    resDesc.res.array.array = cu_array; 
    //object params of the texture 
    struct cudaTextureDesc texDesc; 
    memset(&texDesc,0,sizeof(texDesc)); 
    texDesc.addressMode[0] = cudaAddressModeClamp;     
    texDesc.addressMode[1] = cudaAddressModeClamp;      
    texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    //texture object creation 
    cudaTextureObject_t texObj = 0; 
    cudaCreateTextureObject(&texObj,&resDesc,&texDesc,NULL); 

    if(returnValue != cudaSuccess) printf("\n Got error while running CUDA API Bind Texture");
    cudaMalloc(&dscaled_data, scaled_h*scaled_w*sizeof(unsigned char) ); 

    dim3 dimBlock(32,32,1);
    dim3 dimGrid(scaled_w/dimBlock.x,scaled_h/dimBlock.y,1);
    printf("\n Launching grid with blocks [%d][%d] ", dimGrid.x,dimGrid.y);

    resized<<<dimGrid,dimBlock>>>(dscaled_data,scaled_w,1/ratio,texObj);
    
    returnValue = (cudaError_t)(returnValue | cudaThreadSynchronize());

	returnValue = (cudaError_t)(returnValue |cudaMemcpy (scaled_data , dscaled_data, scaled_h*scaled_w*sizeof(unsigned char), cudaMemcpyDeviceToHost ));
        if(returnValue != cudaSuccess) printf("\n Got error while running CUDA API kernel");

    //destroy texture object
    cudaDestroyTextureObject(texObj);
    scr_write_pgm(outputStr,scaled_data,scaled_h,scaled_w,"####"); //storing image with detections 
    
    if(data!=NULL)
        free(data); 
    if(cu_array!=NULL)
        cudaFreeArray(cu_array); 
    if(scaled_data!=NULL)
        free(scaled_data); 
    if(dscaled_data!=NULL)
        free(dscaled_data); 
 
    return 0; 

}