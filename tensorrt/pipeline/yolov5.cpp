#include <stdio.h>
#include <iostream>
#include <string.h>
#include <cuda_runtime_api.h>
#include "logging.h"
#include "common.hpp"
#include <chrono>



#define USE_FP32  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.5
#define CONF_THRESH 0.4
#define BATCH_SIZE 1

#define NET s  // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)
// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;




class YOLO_INF
{
public:
  
  std::string engine_name ;
  //static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
  //static float prob[BATCH_SIZE * OUTPUT_SIZE];
  nvinfer1::IRuntime* runtime;
  nvinfer1::ICudaEngine* engine;
  char *trtModelStream{ nullptr };
  size_t size{ 0 };
  nvinfer1::IExecutionContext* context;
  cudaStream_t stream;
  void* buffers[2];
  int inputIndex ;
  int outputIndex ;

  YOLO_INF() {
    	  // load engine nvinfer1::weights 
        cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    engine_name = "yolov5s.engine";

	  std::ifstream file(engine_name, std::ios::binary);

        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
             }

    // prepare input data ---------------------------
    runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CHECK(cudaStreamCreate(&stream));


  }

  ~YOLO_INF(){
 // Release stream and buffers 
    cudaStreamDestroy(stream); 
    CHECK(cudaFree(buffers[inputIndex])); 
    CHECK(cudaFree(buffers[outputIndex])); 
    // Destroy the engine 
    context->destroy(); 
    engine->destroy();
    runtime->destroy();
  


  }

void doInference(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void imginfer(const std::string&dir)
{
std::vector<std::string> names = {"samples/img1.jpg","samples/img2.jpg","samples/img3.jpg","samples/img4.jpg","samples/img5.jpg","samples/img6.jpg","samples/img7.jpg"};

//auto a = read_files_in_dir(dir.c_str(),names);
/*
if(a==-1)
{
	std::cout << "No files in " << dir.c_str() << std::endl;
        return;
}
*/
int i = 0; 
for(auto name:names)
{
    all(name,i);
i++;
}
}

int all(const std::string &img_name, const int &num)
{
      // prepare input data ---------------------------
   
        auto start = std::chrono::system_clock::now();
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
              
    
        // Run inference
    cv::Mat img = cv::imread("/home/cedric/Learning_Cuda/tensorrt/pipeline/" + img_name);
            if (img.empty()) 
	    {    std::cout << "img is empty " << std::endl;
		    return 0;}
             
        auto end = std::chrono::system_clock::now();
        std::cout << "img load" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
            
        start = std::chrono::system_clock::now();
	    cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    data[ i] = (float)uc_pixel[2] / 255.0;
                    data[ i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                    data[ i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                    ++i;
                }
            }

        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        std::vector<std::vector<Yolo::Detection>> batch_res(1);
        
    	auto& res = batch_res[0];
            nms(res, &prob[0], CONF_THRESH, NMS_THRESH);
            
        end = std::chrono::system_clock::now();
        std::cout << "YOLO inf" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
            //std::cout << res.size() << std::endl;
          /*    
	cv::Mat img2 = cv::imread("/home/cedric/Learning_Cuda/tensorrt/pipeline/" + img_name);
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
		cv::rectangle(img2, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img2, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite("/home/cedric/Learning_Cuda/tensorrt/pipeline/output/img" + std::to_string(num) + ".jpg", img2);
*/
}

};

