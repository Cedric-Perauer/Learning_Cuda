cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++14;-g;-G;-gencode;arch=compute_30;code=sm_30)

SET("Torch_DIR" "/home/cedric/libtorch/share/cmake/Torch")

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(Torch REQUIRED)
find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5.cpp)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 "${TORCH_LIBRARIES}")
target_link_libraries(yolov5 ${OpenCV_LIBS})
set_property(TARGET yolov5 PROPERTY CXX_STANDARD 14)

add_definitions(-O2 -pthread)

