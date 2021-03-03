#include <torch/torch.h>
#include <iostream>
#include <memory>


struct Resnet: torch::nn::Module {


	Resnet(int64_t in_channels,int64_t  out_channels) : 

                
		conv1(register_module("conv1",torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels,out_channels,3).stride(1).padding(2).dilation(2)))), 
                conv2(register_module("conv2",torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels,out_channels,3).stride(1).padding(1)))), 
		bn(register_module("bn", torch::nn::BatchNorm2d(out_channels))), 
		shortcut_conv(register_module("shortcut_conv",torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels,out_channels,1).stride(1)))),
                relu(register_module("relu",torch::nn::ReLU())) 
		
		{}
        
        torch::Tensor forward(torch::Tensor x) {
                      torch::Tensor x_forward; 
                      x_forward = relu(bn(conv1(x))); 
                      x_forward = relu(bn(conv2(x_forward)));

		      x = relu(bn(shortcut_conv(x)));
                      
		      return relu(x + x_forward); 
	}
 
  torch::nn::Conv2d conv1, conv2, shortcut_conv; 
        torch::nn::BatchNorm2d bn; 
        torch::nn::ReLU relu; 
        torch::nn::LeakyReLU leaky_relu; 



};



struct Resnet_C_Block: torch::nn::Module {
        torch::nn::Conv2d conv1, conv2,conv3;
        torch::nn::BatchNorm2d bn1,bn2,bn3;
        torch::nn::ReLU relu;
        torch::nn::LeakyReLU leaky_relu;





        Resnet_C_Block(int in_channels, int out_channels) :


                conv1(register_module("conv1",torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels,out_channels/4.0,3).padding(1)))),
                conv2(register_module("conv2",torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels/4.0,out_channels/2.0,3).padding(1)))),
                conv3(register_module("conv3",torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels/2.0,out_channels,3).padding(1)))),
                bn1(register_module("bn1", torch::nn::BatchNorm2d(out_channels/4.0))),
                bn2(register_module("bn2", torch::nn::BatchNorm2d(out_channels/2.0))),
                bn3(register_module("bn3", torch::nn::BatchNorm2d(out_channels))),
                relu(register_module("relu",torch::nn::ReLU()))

                {}

        torch::Tensor forward(torch::Tensor x) {
                      x = relu(bn1(conv1(x)));
                      x = relu(bn2(conv2(x)));
                      x = relu(bn3(conv3(x)));


                      return x;
        }



};




struct KeypointNet : torch::nn::Module {
              Resnet res1,res2,res3,res4; 
              Resnet_C_Block res_c; 
              torch::nn::BatchNorm2d bn;
	      torch::nn::Conv2d out; 
              torch::nn::ReLU relu;
	      int net_size = 16;       
	      int num_kpt = 7; 
      KeypointNet() :
               
     //register_module("res1",std::make_shared<Resnet>(net_size,net_size));

	      res2(register_module("res2",Resnet(net_size,2*net_size))),
                res3(register_module("res3",Resnet(net_size*2,4*net_size))),
                res4(register_module("res4",Resnet(net_size*4,8*net_size))),
                
                bn(register_module("bn", torch::nn::BatchNorm2d(net_size))),
                res_c(register_module("res_c",Resnet_C_Block(3,net_size))),
		out(register_module("out",torch::nn::Conv2d(torch::nn::Conv2dOptions( 8 * net_size,num_kpt,1).stride(1).padding(0)))),
                 
                relu(register_module("relu",torch::nn::ReLU()))
		{}

     torch::Tensor forward(torch::Tensor x) {
       x = relu(bn(res_c(x)));
       x = res1(x); 
       x = res2(x); 
       x = res3(x); 
       x = res4(x);
       x = out(x);  


     }
};
