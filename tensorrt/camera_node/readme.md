## Mono Camera Pipeline Template for the Formula Student Driverless Competition 

![](https://github.com/Cedric-Perauer/Learning_Cuda/blob/master/tensorrt/camera_node/Screenshot%20from%202021-02-16%2021-42-20.png "Pipeline running in the FSD Simulator")


### Requirements : 
- CUDA, CuDNN, TenosrRT, Torchscript, ROS (Melodic prefered), OpenCV 

### Intro

- this project runs in ROS (implemented using Melodic) and can be built with the normal caktin commands
- please refer to the [CMAKELists.txt](https://github.com/Cedric-Perauer/Learning_Cuda/blob/master/tensorrt/camera_node/CMakeLists.txt) to update the Torchscript path accordingly
- for using the algorithm on a different camera image topic you can change [this line](https://github.com/Cedric-Perauer/Learning_Cuda/blob/4887d4b6dbcbcf1633b3ac00266d023f75b75f03/tensorrt/camera_node/src/camera_node.cpp#L348) in src/camera_node.cpp
- modify some of the camera parameters and object priors and configure the weights path [here](https://github.com/Cedric-Perauer/Learning_Cuda/blob/4887d4b6dbcbcf1633b3ac00266d023f75b75f03/tensorrt/camera_node/src/camera_node.cpp#L23)
- some camera parameters can also be adjusted in the params [file](https://github.com/Cedric-Perauer/Learning_Cuda/blob/master/tensorrt/camera_node/src/params.txt), 
you may add new ones by refering to the [params.h file](https://github.com/Cedric-Perauer/Learning_Cuda/blob/master/tensorrt/camera_node/src/params.h)

### Open Source projects used

- the implementation is inspired by [MIT Driverless](https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra/tree/master/CVC-YOLOv3) and uses a 
[YOLOv5 object detector](https://github.com/ultralytics/yolov5) running in TensorRT by using the implementation of YOLOv5 in TensorRT by [Wang Xinyu](https://github.com/wang-xinyu/tensorrtx) 
- weights files are not provided here, but I used the [FSOCO dataset](https://www.fsoco-dataset.com) for training the object detector, and the [MIT Dataset](https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra/tree/master/CVC-YOLOv3) for training the keypoint extraction (see points inside the bounding boxes of the image above)

