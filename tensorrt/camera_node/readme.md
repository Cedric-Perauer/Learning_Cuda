## Mono Camera Pipeline Template for the Formula Student Driverless Competition 

![]()


- this project runs in ROS (implemented using Melodic) and can be built with the normal caktin commands
- please refer to the [CMAKELists.txt](https://github.com/Cedric-Perauer/Learning_Cuda/blob/master/tensorrt/camera_node/CMakeLists.txt) to update the Torchscript path accordingly
- for using the algorithm on a different camera image topic you can change [this line](https://github.com/Cedric-Perauer/Learning_Cuda/blob/4887d4b6dbcbcf1633b3ac00266d023f75b75f03/tensorrt/camera_node/src/camera_node.cpp#L348) in src/camera_node.cpp

- the implementation is inspired by [MIT Driverless](https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra/tree/master/CVC-YOLOv3) and uses a 
[YOLOv5 object detector](https://github.com/ultralytics/yolov5) running in TensorRT by using the implementation of YOLOv5 in TensorRT by [Wang Xinyu](https://github.com/wang-xinyu/tensorrtx) 
- weights files are not provided here, but I used the [FSOCO dataset](https://www.fsoco-dataset.com) for training the object detector, and the MIT Dataset for training the keypoint extraction (see points inside the bounding boxes of the image above)
