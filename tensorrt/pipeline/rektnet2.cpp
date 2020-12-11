#include <iostream>
#include "torch/torch.h"
#include "torch/script.h"
#include <memory>

const int REKT_SIZE = 80; 

class Rektnet { 
   torch::jit::script::Module module;
   std::vector<torch::jit::IValue> inputs;
   torch::Tensor tharray; 
   
   public :

   Rektnet(const std::string &name){ 
	   inputs = {}; 
                try {
      module = torch::jit::load("/home/cedric/torch_test/traced_rektnet.pt");
      }

     catch (const c10::Error &e) {
           std::cerr << "error loading module\n";
     }
     tharray = torch::zeros({10,3,80,80}).to(torch::Device("cuda:0")); //or use kF64  
     module.to(torch::Device("cuda:0"));  
   }

  ~Rektnet(){}  
  
  cv::Mat prep_image(const cv::Mat &src)
	{
	   cv::Mat dst;
	   cv::Size size(80,80);
           cv::resize(src,dst,size); 	   
           return dst;                      
	}


  void forward(const std::vector<cv::Mat> &imgs)
  { 
          int bs= 10; 
	  float data[bs][3][REKT_SIZE][REKT_SIZE];
          inputs = {};            

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
	  //std::memcpy(tharray.data_ptr(),data,sizeof(float)*tharray.numel());  
	  
	  inputs.push_back(tharray);
	  auto output = module.forward(inputs).toTensor();	  
  }  
}; 


