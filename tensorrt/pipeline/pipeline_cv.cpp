#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "yolov5.cpp"
#include "rektnet.cpp"
#include <torch/torch.h>


void soft_argmax(float *softmax){ 

int arrSize = sizeof(softmax)/sizeof(softmax[0]);
std::cout << "softmax size: " << arrSize << std::endl;
//auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, 1);
//torch::Tensor tharray = torch::from_blob(array, {5}, options);


} 




int main(int argc, char** argv) {
     
     //inits
     YOLO_INF yolov5 = YOLO_INF();
     Rektnet rektnet = Rektnet(1);
     
     yolov5.all("samples/img1.jpg",1); 
      rektnet.inference("samples_rekt/cut.jpg");  
      auto start = std::chrono::system_clock::now(); 
     yolov5.all("samples/img1.jpg",2); 
    rektnet.inference("samples_rekt/cut.jpg");  
       auto end = std::chrono::system_clock::now();               
      
 std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
     return 0;
}



