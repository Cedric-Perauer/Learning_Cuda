 
#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "yolov5.cpp"
#include "rektnet.cpp"
#include <torch/torch.h>

class Pipeline { 

torch::Tensor values_x; 
torch::Tensor values_y; 
torch::Tensor tharray; 

public : 

Pipeline(){ 
auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU,0);
values_x = torch::linspace(0,(REKT_SIZE -1.0)/REKT_SIZE, REKT_SIZE, options); 
values_y = torch::linspace(0,(REKT_SIZE -1.0)/REKT_SIZE, REKT_SIZE, options); 
tharray = torch::zeros({10,7,80,80},torch::kFloat32); //or use kF64

} 

~Pipeline(){}

torch::Tensor flat_softmax(OUT input) {


auto start = std::chrono::system_clock::now();
//convert array to torch tensor 
std::memcpy(tharray.data_ptr(),input,sizeof(float)*tharray.numel());
torch::Tensor flat = tharray.view({-1,REKT_SIZE*REKT_SIZE}); 
flat = torch::nn::functional::softmax(flat,1);
torch::Tensor hm = flat.view({-1,7,REKT_SIZE,REKT_SIZE}); 
torch::Tensor exp_y = (hm.sum(3) * values_y).sum(-1); 
torch::Tensor exp_x = (hm.sum(2) * values_x).sum(-1); 


auto end = std::chrono::system_clock::now();
    std::cout << "Softmax " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

return torch::stack({exp_x, exp_y},-1); 
} 

void plot_pts(std::vector<Yolo::Detection>& res, std::vector<std::vector<float>> &bbox_vals, torch::Tensor &rektnet)
{

cv::Mat img2 = cv::imread("/home/pjfsd/Learning_Cuda/tensorrt/pipeline/samples/img3.jpg");
for(size_t i=0; i < 10;i++)
{ 
          cv::Rect r = get_rect(img2, res[i].bbox);
	  cv::rectangle(img2, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
          cv::putText(img2, std::to_string((int)res[i].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
	  
	  for(int j = 0; j < 7 ;++j)
	  {    
		  std::vector<float> v(rektnet[i][j].data_ptr<float>(), rektnet[i][j].data_ptr<float>() + rektnet[i][j].numel());
		  cv::circle(img2,cv::Point(r.x+int(v[0] * r.width), r.y + int(v[1] * r.height)),2,cv::Scalar(255,0,0),3);  

	  }


}	

cv::imshow("circle",img2); 
cv::waitKey(0); 


}


}; 



int main(int argc, char** argv) 
{
     
     //inits
     YOLO_INF yolov5 = YOLO_INF();
     Rektnet rektnet = Rektnet(10);
     Pipeline pipeline = Pipeline(); 
     
     std::vector<cv::Mat> imgs = yolov5.inference("samples/img3.jpg",1); 
     for(int i= 0; i < 10; ++i)
     { 
     auto start = std::chrono::system_clock::now();

     imgs = yolov5.inference("samples/img3.jpg",1); 
      
     auto out_rekt = rektnet.inference(imgs);  
     auto out = pipeline.flat_softmax(out_rekt); 
     auto end = std::chrono::system_clock::now();
    std::cout << "total time :" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
     pipeline.plot_pts(yolov5.res_sorted,yolov5.box_coords,out ); 
     }
     return 0;
}
