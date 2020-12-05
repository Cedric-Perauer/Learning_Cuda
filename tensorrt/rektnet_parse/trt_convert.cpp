#include <string.h>
#include <stdio.h>
#include <iostream>
#include "cuda_runtime_api.h"
#include "logging.h"

#define USE_FP16
const int BATCH_SIZE = 10; 


int main(int argc, char ** argv) 
{
static Logger gLogger; 
nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);
auto parser = nvonnxparser::createParser(*network, gLogger);


builder->setMaxBatchSize(BATCH_SIZE); 
config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;





return 0;
} 
