#include <iostream>
#include <chrono>
//#include "logging.h"
#include "common.hpp"
#include "yolov5.cpp"
#include <iostream>
#include <memory>
#include "rektnet2.cpp"




int main(int argc, char** argv) {
     //inits
     std::cout << torch::cuda::is_available() << std::endl;
      auto start = std::chrono::system_clock::now(); 
      auto end = std::chrono::system_clock::now(); 
     YOLO_INF yolov5 = YOLO_INF();
     std::vector<cv::Mat> imgs; 
     Rektnet rektnet = Rektnet("/home/cedric/torch_test/traced_rektnet.pt"); 
     
     for(int i = 0; i < 10; ++i) {
     start = std::chrono::system_clock::now();
	     imgs = yolov5.inference("samples/img1.jpg",1); 
     std::cout <<"yolo done" << std::endl;
	     rektnet.forward(imgs);
    end = std::chrono::system_clock::now();
std::cout << "Total run " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

     }  
     
     
     return 0;
}



