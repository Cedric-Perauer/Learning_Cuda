#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "yolov5.cpp"
#include "rektnet.cpp"
#include <torch/torch.h>


torch::Tensor flat_softmax(OUT input) {

auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU,0);
torch::Tensor values_x = torch::linspace(0,(REKT_SIZE -1.0)/REKT_SIZE, REKT_SIZE, options); 
torch::Tensor values_y = torch::linspace(0,(REKT_SIZE -1.0)/REKT_SIZE, REKT_SIZE, options); 
auto tharray = torch::zeros({10,7,80,80},torch::kFloat32); //or use kF64

auto start = std::chrono::system_clock::now();
//convert array to torch tensor 
std::memcpy(tharray.data_ptr(),input,sizeof(float)*tharray.numel());
torch::Tensor flat = tharray.view({-1,REKT_SIZE*REKT_SIZE}); 
flat = torch::nn::functional::softmax(flat,1);
torch::Tensor hm = flat.view({-1,7,REKT_SIZE,REKT_SIZE}); 
torch::Tensor exp_y = (hm.sum(3) * values_y).sum(-1); 
torch::Tensor exp_x = (hm.sum(2) * values_x).sum(-1); 


auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

return torch::stack({exp_x, exp_y},-1); 
}



int main(int argc, char** argv) {
     
     //inits
     YOLO_INF yolov5 = YOLO_INF();
     Rektnet rektnet = Rektnet(10);
     
      auto out_rekt = rektnet.inference("samples_rekt/cut.jpg");  
      auto start = std::chrono::system_clock::now();
      auto out = flat_softmax(out_rekt); 
     std::cout << "size" << out.sizes() << std::endl; 
      //torch::Tensor out = soft_argmax(out_rekt); 
      std::cout << out << std::endl;
     auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;    
      
    return 0;
}



