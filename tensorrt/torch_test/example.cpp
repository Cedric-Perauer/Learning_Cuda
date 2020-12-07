#include <iostream>
#include "torch/torch.h"
#include "torch/script.h"
#include <memory>

int main() { 
 torch::jit::script::Module module;

        try {
      module = torch::jit::load("/home/cedric/torch_test/traced_rektnet.pt");     
     }

     catch (const c10::Error &e) {
           std::cerr << "error loading module\n";
           return -1;
     }
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::ones({10, 3, 80, 80}).to(torch::Device("cuda:0")));

module.to(torch::Device("cuda:0"));
auto output = module.forward(inputs).toTensor();

std::cout << output << std::endl;
//auto end = std::chrono::system_clock::now();
   //  std::cout << output << std::endl;

  //  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
return 0;

} 
