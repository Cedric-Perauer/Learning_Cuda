## Mono Camera Pipeline Template for the Formula Student Driverless Competition 

![](https://github.com/Cedric-Perauer/Learning_Cuda/blob/master/tensorrt/camera_node/Screenshot%20from%202021-02-16%2021-42-20.png "Pipeline running in the FSD Simulator")


### Requirements : 
- CUDA, CuDNN, TenosrRT, Torchscript, ROS (Melodic prefered), OpenCV 

### Intro

- this project runs in ROS (implemented using Melodic) and can be built with the normal caktin commands
- please refer to the [CMAKELists.txt](https://github.com/Cedric-Perauer/Learning_Cuda/blob/master/tensorrt/camera_node/CMakeLists.txt) to update the Torchscript path accordingly
- for using the algorithm on a different camera image topic you can change [this line](https://github.com/Cedric-Perauer/Learning_Cuda/blob/4887d4b6dbcbcf1633b3ac00266d023f75b75f03/tensorrt/camera_node/src/camera_node.cpp#L348) in src/camera_node.cpp
- the keypoint extractor weights may be changed here [](https://github.com/Cedric-Perauer/Learning_Cuda/blob/e80700f8f86874c4d1c5b725d95df2778c49394b/tensorrt/camera_node/src/rektnet.cpp#L40)
- modify some of the camera parameters and object priors and configure the weights path of the object detector [here](https://github.com/Cedric-Perauer/Learning_Cuda/blob/4887d4b6dbcbcf1633b3ac00266d023f75b75f03/tensorrt/camera_node/src/camera_node.cpp#L23)
- some camera parameters can also be adjusted in the params [file](https://github.com/Cedric-Perauer/Learning_Cuda/blob/master/tensorrt/camera_node/src/params.txt), 
you may add new ones by refering to the [params.h file](https://github.com/Cedric-Perauer/Learning_Cuda/blob/master/tensorrt/camera_node/src/params.h)
- since the pipeline can not handle occluded objects, objects inside the car mask or at the edge are not passed on, you may change these masks [here](https://www.researchgate.net/publication/349929458_Bachelor_Thesis_Development_and_Deployment_of_a_Perception_Stack_for_the_Formula_Student_Driverless_Competition) and [here](https://github.com/Cedric-Perauer/Learning_Cuda/blob/e80700f8f86874c4d1c5b725d95df2778c49394b/tensorrt/camera_node/src/camera_node.cpp#L84)
- for a full explanation of the pipeline refer to the thesis [here](https://www.researchgate.net/publication/349929458_Bachelor_Thesis_Development_and_Deployment_of_a_Perception_Stack_for_the_Formula_Student_Driverless_Competition) 


### Open Source projects used

- the implementation is inspired by [MIT Driverless](https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra/tree/master/CVC-YOLOv3) and uses a 
[YOLOv5 object detector](https://github.com/ultralytics/yolov5) running in TensorRT by using the implementation of YOLOv5 in TensorRT by [Wang Xinyu](https://github.com/wang-xinyu/tensorrtx) 
- weights files are not provided here, but I used the [FSOCO dataset](https://www.fsoco-dataset.com) for training the object detector, and the [MIT Dataset](https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra/tree/master/CVC-YOLOv3) for training the keypoint extraction (see points inside the bounding boxes of the image above)

