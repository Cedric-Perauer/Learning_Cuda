#include <torch/torch.h>
#include <iostream>



struct Resnet: torch::nn::Module {
        torch::nn::Conv2d conv1, conv2, shortcut_conv; 
        torch::nn::BatchNorm2d bn; 
        torch::nn::ReLU relu; 
        torch::nn::LeakyReLU leaky_relu; 





	Resnet(in_channels, out_channels) : 

                
		conv1(register_module("conv1",torch::nn::Conv2d(in_channels,out_channels,3,1,2,2))); 
                conv2(register_module("conv2",torch::nn::Conv2d(out_channels,out_channels,3,1,1))); 
		bn(register_module("bn", torch::nn::BatchNorm2d(out_channels))); 
		shortcut_conv(register_module("shortcut_conv",torch::nn::Conv2d(in_channels,out_channels,1,1)));
                relu(register_module("relu",torch::nn::ReLU())); 
		
	}
        
        torch::Tensor forward(torch::Tensor x) {
                      torch::Tensor x_forward; 
                      x_forward = relu(bn(conv1(x))); 
                      x_forward = relu(bn(conv2(x_forward)));

		      x = relu(bn(shortcut_conv(x)));
                      
		      return relu(x + x_forward); 
	}
 


}

