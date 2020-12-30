#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "yolov5.cpp"
#include "rektnet.cpp"
#include "torch/torch.h"
#include <sstream> 

const float FOCAL_LENGTH = 1.2e-3; 
const float CONE_HEIGHT = 0.3;
const float PIXEL_SIZE = 6e-6; 
const int IMG_HALF = 800; 

using namespace torch::indexing; 

class Pipeline { 

torch::Tensor values_x; 
torch::Tensor values_y; 
torch::Tensor tharray; 
//masks for the car and the image edge 
std::vector<std::vector<cv::Point>> car_coordinates; 
std::vector<std::vector<cv::Point>> edge_coordinates; 
//tensor options 
torch::TensorOptions options; 
torch::Tensor tharray_gpu; //tharray stores REKTNET output 
torch::Tensor slopes; //slopes for mid point for height computation 
torch::Tensor yolo_out; //yolo outputs include box position and height/width for 3D pose estimation  
torch::Tensor yolo_out_gpu; //GPU yolo outputs include box position and height/width for 3D pose estimation  

public : 

Pipeline(){

//tensors and tensor options 	
options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA,0);
values_x = torch::linspace(0,(REKT_SIZE -1.0)/REKT_SIZE, REKT_SIZE, options); 
values_y = torch::linspace(0,(REKT_SIZE -1.0)/REKT_SIZE, REKT_SIZE, options); 
tharray = torch::zeros({BATCH_SIZE_REKT,7,80,80},torch::kFloat32).to(torch::kCPU); //or use kF64
tharray_gpu = torch::zeros({BATCH_SIZE_REKT,7,80,80},torch::kFloat32).to(torch::kCUDA); 
yolo_out = torch::zeros({BATCH_SIZE_REKT,5},torch::kFloat32).to(torch::kCPU);  
yolo_out_gpu = torch::zeros({BATCH_SIZE_REKT,5},torch::kFloat32).to(torch::kCUDA);  



//masks
car_coordinates.push_back(std::vector<cv::Point>()); 
//car mask 
car_coordinates[0].push_back(cv::Point(40,1200)); 
car_coordinates[0].push_back(cv::Point(54,1180)); 
car_coordinates[0].push_back(cv::Point(291,1129)); 
car_coordinates[0].push_back(cv::Point(630,1095)); 
car_coordinates[0].push_back(cv::Point(696,983)); 
car_coordinates[0].push_back(cv::Point(728,962)); 
car_coordinates[0].push_back(cv::Point(811,938)); 
car_coordinates[0].push_back(cv::Point(845,937)); 
car_coordinates[0].push_back(cv::Point(843,883)); 
car_coordinates[0].push_back(cv::Point(855,883)); 
car_coordinates[0].push_back(cv::Point(858,942)); 
car_coordinates[0].push_back(cv::Point(947,959)); 
car_coordinates[0].push_back(cv::Point(979,969)); 
car_coordinates[0].push_back(cv::Point(996,982)); 
car_coordinates[0].push_back(cv::Point(1121,1091)); 
car_coordinates[0].push_back(cv::Point(1186,1113)); 
car_coordinates[0].push_back(cv::Point(1292,1113)); 
car_coordinates[0].push_back(cv::Point(1431,1131)); 
car_coordinates[0].push_back(cv::Point(1600,1135)); 
car_coordinates[0].push_back(cv::Point(1600,1200)); 
//edge mask
edge_coordinates.push_back(std::vector<cv::Point>()); 
edge_coordinates[0].push_back(cv::Point(0,300)); 
edge_coordinates[0].push_back(cv::Point(0,300)); 
edge_coordinates[0].push_back(cv::Point(0,1200)); 
edge_coordinates[0].push_back(cv::Point(1600,1200)); 
edge_coordinates[0].push_back(cv::Point(1600,300)); 
edge_coordinates[0].push_back(cv::Point(1580,300)); 
edge_coordinates[0].push_back(cv::Point(1580,1180)); 
edge_coordinates[0].push_back(cv::Point(20,1180)); 
edge_coordinates[0].push_back(cv::Point(20,300)); 

slopes = torch::ones({BATCH_SIZE_REKT}).to(torch::kCUDA);  
} 

~Pipeline(){}

torch::Tensor flat_softmax(OUT input) {

auto start = std::chrono::system_clock::now();
//convert array to torch tensor and move it to the GPU 
std::memcpy(tharray.data_ptr(),input,sizeof(float)*tharray.numel());
tharray_gpu = tharray.to(torch::kCUDA); 
torch::Tensor flat = tharray_gpu.view({-1,REKT_SIZE*REKT_SIZE}); 
flat = torch::nn::functional::softmax(flat,1);
torch::Tensor hm = flat.view({-1,7,REKT_SIZE,REKT_SIZE}); 
torch::Tensor exp_y = (hm.sum(3) * values_y).sum(-1); 
torch::Tensor exp_x = (hm.sum(2) * values_x).sum(-1); 


auto end = std::chrono::system_clock::now();
    std::cout << "Softmax " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

return torch::stack({exp_x, exp_y},-1); 
} 




torch::Tensor distance_calculate(const torch::Tensor &data_batch, float (*input)[5] ) {
     
     auto start = std::chrono::system_clock::now(); //only time based 
     
     //heights of bounding boxes to Tensor  
     std::memcpy(yolo_out.data_ptr(),input,sizeof(float)*yolo_out.numel());
     yolo_out_gpu = yolo_out.to(torch::kCUDA); 
     
     //compute slope for midpoint 
     torch::Tensor slope = -1 * (data_batch.index({"...",6,1}) - data_batch.index({"...",5,1}))/(data_batch.index({"...",6,0}) - data_batch.index({"...",5,0})); 
     //max(0,slope) 
     torch::Tensor relu_data = torch::relu(slope);
     torch::Tensor t = at::nonzero(relu_data);
     slopes = slopes.index_put_({t},-1);

     //midpoint components
     torch::Tensor mid1 = data_batch.index({"...",6,Slice(0,2)}) - data_batch.index({"...",5,Slice(0,2)}); 
     torch::Tensor mid2 = torch::ones({mid1.sizes()[0],2}).to(torch::kCUDA) * 0.5 * torch::stack({slopes,slopes},1);//change slope right here
     torch::Tensor mid3 = data_batch.index({"...",5,Slice(0,2)});
     
     //compute midpoint 
     torch::Tensor mid_pts = mid1 * mid2 + mid3;  
     //----------------------------------TO-DO----------------------------CHANGE HEIGHT of 123---------

     torch::Tensor top_pt = (mid_pts - data_batch.index({"...",0,Slice(0,2)}));  
     torch::Tensor img_h = torch::norm(top_pt,2,1) *  yolo_out_gpu.index({"...",3}); 
     torch::Tensor hbb = img_h * PIXEL_SIZE; 
     
     //compute distances 
     torch::Tensor distances = FOCAL_LENGTH / (hbb * CONE_HEIGHT);  	 
     

     //compute x pose ------------------------------------------------------
     torch::Tensor bbox_xcenter = PIXEL_SIZE * (yolo_out_gpu.index({"...",2}) - IMG_HALF); //compute x center pose as offset from middle line of the image 
     torch::Tensor widths = (bbox_xcenter * distance) / FOCAL_LENGTH; //compute x pose in 3D space 


    //time based 
     auto end = std::chrono::system_clock::now();
     std::cout << "total time depth estimation :" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
     std::cout << "distances " << distances << std::endl; 

     //return distances 
     return distances;  

     //uncomment this after fixing plot function and so on 
     //return stacked x,y poses in 3D space
     //return torch::stack({distances,widths},1); 
} 

void plot_pts(std::vector<Yolo::Detection>& res, std::vector<std::vector<float>> &bbox_vals, torch::Tensor &rektnet, torch::Tensor &distances)
{
torch::Tensor dist = distances.to(torch::kCPU);
cv::Mat img2 = cv::imread("/home/pjfsd/Learning_Cuda/tensorrt/pipeline/samples/img3.jpg");
for(size_t i=0; i < BATCH_SIZE_REKT;i++)
{ 
        
         std::vector<float> d(dist[i].data_ptr<float>(), dist[i].data_ptr<float>() + dist[i].numel());	
       	cv::Rect r = get_rect(img2, res[i].bbox);
	  cv::rectangle(img2, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
          //cv::putText(img2, std::to_string((int)res[i].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
	  std::ostringstream ss;
	  ss << d[0];
	  cv::putText(img2, ss.str(), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
          
	  std::cout << rektnet.sizes() << std::endl; 
	  rektnet = rektnet.to(torch::kCPU); 
	  for(int j = 0; j < 7 ;++j)
	  {    
		  std::vector<float> v(rektnet[i][j].data_ptr<float>(), rektnet[i][j].data_ptr<float>() + rektnet[i][j].numel());
		  cv::circle(img2,cv::Point(r.x+int(v[0] * r.width), r.y + int(v[1] * r.height)),2,cv::Scalar(255,0,0),3);  

	  }

}	
cv::drawContours(img2,car_coordinates,0, cv::Scalar(255,0,0),cv::FILLED, 8 );
cv::drawContours(img2,edge_coordinates,0, cv::Scalar(255,0,0),cv::FILLED, 8 );


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
     for(int i= 0; i < 100; ++i)
     { 
     auto start = std::chrono::system_clock::now();

     imgs = yolov5.inference("samples/img3.jpg",1); 
      
     auto out_rekt = rektnet.inference(imgs);  
     auto out = pipeline.flat_softmax(out_rekt); 
     auto dist = pipeline.distance_calculate(out,yolov5.boxes); 
     auto end = std::chrono::system_clock::now();
     std::cout << "total time :" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
     pipeline.plot_pts(yolov5.res_sorted,yolov5.box_coords,out,dist ); 
     }
     return 0;
}
