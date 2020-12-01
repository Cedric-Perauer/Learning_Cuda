#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "yolov5.cpp"
#include "rektnet.cpp"
#include <torch/torch.h>


torch::Tensor soft_argmax(OUT input){ 

auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU,0);

auto tharray = torch::zeros((1,7,80,80),torch::kFloat32); //or use kF64
std::memcpy(tharray.data_ptr(),input,sizeof(float)*tharray.numel());
torch::Tensor values_x = torch::linspace(0,(REKT_SIZE -1.0)/REKT_SIZE, REKT_SIZE, options); 
torch::Tensor values_y = torch::linspace(0,(REKT_SIZE -1.0)/REKT_SIZE, REKT_SIZE, options); 

std::cout << "shapes" << tharray.sizes()  << std::endl;
//torch::Tensor exp_y = (tharray.sum(2) * values_y).sum(-1); 
//torch::Tensor exp_x = (tharray.sum(2) * values_x).sum(-1); 

//return torch::stack({exp_x,exp_y});
return values_x; 
} 




int main(int argc, char** argv) {
     
     //inits
     YOLO_INF yolov5 = YOLO_INF();
     Rektnet rektnet = Rektnet(1);
     
      auto out_rekt = rektnet.inference("samples_rekt/cut.jpg");  
       auto start = std::chrono::system_clock::now();
      torch::Tensor out = soft_argmax(out_rekt); 
 auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;    
 return 0;
}



