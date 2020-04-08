############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(cs344)

find_package(CUDA REQUIRED)
add_definitions(-std=c++11)


set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  We tell it to support most devices, though, 
  # to make sure more people can easily run class code without knowing 
  # about this compiler argument
  set(CUDA_NVCC_FLAGS "
  -gencode;arch=compute_30,code=sm_30;  
  -gencode;arch=compute_35,code=sm_35;
  -gencode;arch=compute_35,code=compute_35; ")

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
    set(CXX_FLAGS "-Wall")
    set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -fPIC")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
  
else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()

	file( GLOB  hdr *.hpp *.h )
	file( GLOB  cu  *.cu)
	SET (HW1_files exercise1.cu)
	SET (HW2_files exercise2.cu)
	SET (EX3_files exercise3.cu)
	CUDA_ADD_EXECUTABLE(HW1 ${HW1_files})
	CUDA_ADD_EXECUTABLE(HW2 ${HW2_files})
	CUDA_ADD_EXECUTABLE(HW3 ${EX3_files})

