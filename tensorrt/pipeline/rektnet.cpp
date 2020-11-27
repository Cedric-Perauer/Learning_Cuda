#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <chrono>


#define SIZE 80 //rektnet input size 
#define DEVICE 0 
static Logger dLogger;
const char* INPUT_BLOB = "data";
const char* OUTPUT_BLOB = "prob";

class Rektnet {
        private : 
	 IRuntime* runtime; 
	 ICudaEngine* engine; 
	 char*trtModelStream{ nullptr };
	 size_t size{ 0 };
	 IExecutionContext* context;
	 cudaStream_t stream;
         std::string engine_name;
         int batchSize; 
         int inputIndex;  
	 int outputIndex;
         void *buffers[2];

	public : 

	Rektnet(const int &batchSize){
	cudaSetDevice(DEVICE);
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
        runtime = createInferRuntime(dLogger);
        assert(runtime != nullptr);
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(engine != nullptr);
        context = engine->createExecutionContext();
        assert(context != nullptr);
        delete[] trtModelStream;
        assert(engine->getNbBindings() == 2);
        
	inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on deviceF
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * SIZE * SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * SIZE*SIZE *7 * sizeof(float)));
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


        cv::Mat prep_image()
	{
	  
           
	}

       
        void create_engine() 
	{
          

	}








}; 
