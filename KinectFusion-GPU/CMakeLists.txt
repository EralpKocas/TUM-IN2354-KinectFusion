cmake_minimum_required(VERSION 3.16)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(KinectFusion-GPU)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA 6.5 REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -Wall")

IF ((CUDA_VERSION VERSION_GREATER "10.0") OR (CUDA_VERSION VERSION_EQUAL "10.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
ENDIF()

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")

set(SOURCE_FILES
        main.cu
        )

#add_executable(KinectFusion-GPU main.cu ${SOURCE_FILES})
file(GLOB KinectFusion_SRCS "./*.cu" "./*.cpp")

cuda_add_library(KinectFusion STATIC "${KinectFusion_SRCS}")
#cuda_add_executable(KinectFusion STATIC "${KinectFusion_SRCS}")
cuda_add_executable(run-kf-gpu main.cu)
#target_link_libraries(run-kf-gpu -lKinectFusion)
