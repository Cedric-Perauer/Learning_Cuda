#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <chrono>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/types.hpp>

#define REKT_DEVICE 0 
static Logger dLogger;
const int REKT_SIZE = 80; 

typedef float (*OUT)[7][REKT_SIZE][REKT_SIZE]; 

class Rektnet {
        private : 
	 nvinfer1::IRuntime* runtime; 
	 nvinfer1::ICudaEngine* engine; 
	 char*trtModelStream{ nullptr };
	 size_t size{ 0 };
	 nvinfer1::IExecutionContext* context;
	 cudaStream_t stream;
         std::string engine_name;
         int inputIndex;  
	 int outputIndex;
         void* buffers[2];
         float data[10][3][REKT_SIZE][REKT_SIZE];
         float pts[10][7][REKT_SIZE][REKT_SIZE];
         int bs; 

	public : 

	Rektnet(const int &batchSize){
        
	bs = batchSize; 	
	cudaSetDevice(REKT_DEVICE);
        engine_name = "rektnet.engine"; 
        std::ifstream file(engine_name,std::ios::binary);
        
        if(file.good()) {
	 file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
             }	
        runtime = nvinfer1::createInferRuntime(dLogger);
        assert(runtime != nullptr);
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(engine != nullptr);
        context = engine->createExecutionContext();
        assert(context != nullptr);
        delete[] trtModelStream;
        assert(engine->getNbBindings() == 2);
    // Create stream
	inputIndex = engine->getBindingIndex("input.1");
        outputIndex = engine->getBindingIndex("160");
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        // Create GPU buffers on device



	CHECK(cudaMalloc(&buffers[inputIndex], batchSize *  3 * REKT_SIZE * REKT_SIZE * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize *  REKT_SIZE * REKT_SIZE * 7 * sizeof(float)));
	// Create stream
        CHECK(cudaStreamCreate(&stream));
	}


	~Rektnet(){
	
	 cudaStreamDestroy(stream); 
         // Destroy the engine 
         context->destroy(); 
         engine->destroy();
         runtime->destroy();
	
	}


        cv::Mat prep_image(const cv::Mat &src)
	{
	   cv::Mat dst;
	   cv::Size size(80,80);
           cv::resize(src,dst,size); 	   
           return dst;                      
	}
        
/*
        void imginfer(const std::string&dir)
	{
	std::vector<std::string> names = {"samples/cut.jpg","samples/cut.jpg","samples/cut.jpg","samples/cut.jpg","samples/cut.jpg","samples/cut.jpg"};
	//auto a = read_files_in_dir(dir.c_str(),names);
	if(a==-1)
	{
		std::cout << "No files in " << dir.c_str() << std::endl;
		return;
	}
	int i = 0; 
	for(auto name:names)
	{
	    inference(name);
	i++;
	}
	}        

 
        OUT inference(const std::string &filename) 
	{
	  cv::Mat img = cv::imread("/home/cedric/Learning_Cuda/tensorrt/pipeline/" +filename); 

          auto start = std::chrono::system_clock::now();
	  cv::Mat pr_img = prep_image(img);
	  int i = 0;
          
          for(int d = 0; d < batchSize; ++d) {
	   
	  for (int row = 0; row < REKT_SIZE; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < REKT_SIZE; ++col) {
                    data[d][0][row][col] = (float)uc_pixel[2] / 255.0;
                    data[d][1][row][col] = (float)uc_pixel[1] / 255.0;
                    data[d][2][row][col] = (float)uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                    ++i;
                }
            }
          }  
	  int data_sz = sizeof(pts)/sizeof(pts[0]);
	  std::cout << "data size" << data_sz << std::endl;
         
	  CHECK(cudaMemcpyAsync(buffers[0], data, 10*  3 * REKT_SIZE * REKT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
          context->enqueue(batchSize, buffers, stream, nullptr);
          CHECK(cudaMemcpyAsync(pts, buffers[1], 10 * REKT_SIZE *REKT_SIZE * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
          cudaStreamSynchronize(stream);
           
         

	  auto end = std::chrono::system_clock::now();
         std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        return pts;  
	
       	} 
*/

OUT inference(const std::vector<cv::Mat> &imgs) 
	{
          
          for(int d = 0; d < bs; ++d) {
	  
          int i = d; 		  
	  if(i >= imgs.size())
	  { 
	  i = imgs.size() -1; 
	  }
	  cv::Mat pr_img = prep_image(imgs[i]);
	  for (int row = 0; row < REKT_SIZE; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < REKT_SIZE; ++col) {
                    data[d][0][row][col] = (float)uc_pixel[2] / 255.0;
                    data[d][1][row][col] = (float)uc_pixel[1] / 255.0;
                    data[d][2][row][col] = (float)uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                }
            }
          }  
         
          cudaEvent_t start, stop;
          cudaEventCreate(&start);
          cudaEventCreate(&stop);
	  
	  CHECK(cudaMemcpyAsync(buffers[0], data, bs*  3 * REKT_SIZE * REKT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
	  cudaEventRecord(start);
	  context->enqueue(bs, buffers, stream, nullptr);
          cudaEventRecord(stop);
	  CHECK(cudaMemcpyAsync(pts, buffers[1], bs * REKT_SIZE *REKT_SIZE * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
          cudaStreamSynchronize(stream);
          cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop); 
       
        std::cout << "kernel execution time " << milliseconds << std::endl;
        return pts;  
	
       	}

}; 
