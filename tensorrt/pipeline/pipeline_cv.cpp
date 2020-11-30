#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "yolov5.cpp"
#include "rektnet.cpp"
#include <torch/torch.h>


int main(int argc, char** argv) {
     
     //inits
     YOLO_INF yolov5 = YOLO_INF();
     Rektnet rektnet = Rektnet(1);
     
     //yolov5.imginfer("../samples/"); 
     rektnet.imginfer("../samples/");  

     return 0;
}






