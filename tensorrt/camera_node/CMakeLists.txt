cmake_minimum_required(VERSION 2.8.3)
project(camera_node)

SET("Torch_DIR" "/home/cedric/libtorch/share/cmake/Torch/")
add_definitions(-std=c++14 -g -lstdc++fs)
set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")


find_package(catkin_simple 0.1.0 REQUIRED )
find_package(Torch REQUIRED)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++14;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/src/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)


##ros specific
catkin_simple(ALL_DEPS_REQUIRED)




find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        nav_msgs
	sensor_msgs
     	cv_bridge
     	image_transport)

catkin_package()

include_directories(
        include
	src
	${CMAKE_CURRENT_SOURCE_DIR}
        ${catkin_INCLUDE_DIRS}
        ${roscpp_INCLUDE_DIRS}
        ${rospy_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS} 
)
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS}) 

add_executable(camera_node src/camera_node.cpp)
target_link_libraries(camera_node ${catkin_LIBRARIES} ${Pylon_LIBRARIES} ${OpenCV_LIBRARIES} )
target_link_libraries(camera_node nvinfer)
target_link_libraries(camera_node cudart)
target_link_libraries(camera_node myplugins)
target_link_libraries(camera_node "${TORCH_LIBRARIES}")
set_property(TARGET camera_node PROPERTY CXX_STANDARD 14)

