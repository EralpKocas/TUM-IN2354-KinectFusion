cmake_minimum_required(VERSION 3.16)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(KinectFusion-GPU)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(CUDA 6.5 REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -Wall")

IF ((CUDA_VERSION VERSION_GREATER "10.0") OR (CUDA_VERSION VERSION_EQUAL "10.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
ENDIF()

#SET("OpenCV_DIR" "~/Documents/opencv/usr/local/share/OpenCV")
#find_package(OpenCV 3.0 REQUIRED)
#if (OpenCV_INCLUDE_DIRS)
#    include_directories("${OpenCV_INCLUDE_DIRS}")
#    link_directories(${OpenCV_DIR}/lib)
#endif (OpenCV_INCLUDE_DIRS)

#set(OpenCV_INCLUDE_DIR /usr/local/include/opencv4/)
#set(OpenCV_LIB_DIR /usr/local/lib/)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(FreeImage_INCLUDE_DIR /usr/include/)
set(FreeImage_LIBRARY_DIR /usr/lib/x86_64-linux-gnu/)

message(${OpenCV_INCLUDE_DIRS})
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")
set(SOURCE_FILES
        main.cu
        data_types.h
        Eigen.h
        common_functions.h
        FreeImageHelper.h
        FreeImageHelper.cpp
        VirtualSensor_freiburg.h)

#add_executable(KinectFusion-GPU main.cu ${SOURCE_FILES})
file(GLOB KinectFusion_SRCS "./*.cu" "./*.cpp" "./*.h")

cuda_add_library(run-kf-gpu-lib STATIC "${KinectFusion_SRCS}")
link_directories(${FreeImage_LIBRARY_DIR})
cuda_add_executable(run-kf-gpu "${SOURCE_FILES}")
target_link_libraries(run-kf-gpu Eigen3::Eigen freeimage ${OpenCV_LIBS})
target_include_directories(run-kf-gpu PUBLIC ${OpenCV_INCLUDE_DIRS})