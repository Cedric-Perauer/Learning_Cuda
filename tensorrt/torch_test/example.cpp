#include <iostream>
#include "torch/torch.h"
#include "torch/script.h"
#include <memory>
#include <chrono>

using namespace torch::indexing; 

const float FOCAL_LENGTH = 1.2e-3; 
const float CONE_HEIGHT = 0.3;
const float PIXEL_SIZE = 6e-6; 

torch::Tensor calc_distances(torch::Tensor data) {
    torch::Tensor data_batch = torch::stack({data,data});
    torch::Tensor slopes = torch::ones({data_batch.sizes()[0]}).to(torch::kCUDA);  
    auto start = std::chrono::system_clock::now(); 
    torch::Tensor slope = -1 * (data_batch.index({"...",6,1}) - data_batch.index({"...",5,1}))/(data_batch.index({"...",6,0}) - data_batch.index({"...",5,0})); 
     torch::Tensor relu_data = torch::relu(slope);
     torch::Tensor t = at::nonzero(relu_data);
     slopes = slopes.index_put_({t},-1);
    
     
     //torch::Tensor mid1 = data_batch.index({"...",Slice(5,7),1}) - data_batch.index({"...",Slice(5,7),1});  
     torch::Tensor mid1 = data_batch.index({"...",6,Slice(0,2)}) - data_batch.index({"...",5,Slice(0,2)}); 
     torch::Tensor mid2 = torch::ones({mid1.sizes()[0],2}).to(torch::kCUDA) * 0.5 * torch::stack({slopes,slopes},1);
     torch::Tensor mid3 = data_batch.index({"...",5,Slice(0,2)});

     torch::Tensor mid_pts = mid1 * mid2 + mid3;  
     torch::Tensor top_pt = (mid_pts - data_batch.index({"...",0,Slice(0,2)})) * 132;  
     std::cout << top_pt << std::endl;
     torch::Tensor img_h = torch::norm(top_pt,2,1); 
     torch::Tensor hbb = img_h * PIXEL_SIZE; 

     std::cout << img_h << std::endl;
     torch::Tensor distances = FOCAL_LENGTH / (hbb * CONE_HEIGHT);  	 
     std::cout << distances << std::endl;
     return distances;
} 


int main() { 
    



    float a[7][2] = {{0.46115, 0.05223},{0.35005, 0.31250},{0.60000, 0.31274},{0.31250, 0.53756},{0.65101, 0.55000},{0.24339, 0.79127},{0.71530, 0.79728}};

    torch::Tensor data= torch::from_blob(a,{7,2}).to(torch::kCUDA);
    auto dist = calc_distances(data);     
    return 0;

} 
