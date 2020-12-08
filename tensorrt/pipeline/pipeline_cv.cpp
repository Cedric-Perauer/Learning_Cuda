#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "yolov5.cpp"
#include "rektnet2.cpp"





int main(int argc, char** argv) {
     //inits
     YOLO_INF yolov5 = YOLO_INF();
     Rektnet rektnet = Rektnet("/home/cedric/torch_test/traced_rektnet.pt"); 
     std::vector<cv::Mat> imgs = yolov5.inference("samples/img1.jpg",1); 
     imgs = yolov5.inference("samples/img1.jpg",1); 
     
     out_rekt = rektnet.inference(imgs);  
     return 0;
}



