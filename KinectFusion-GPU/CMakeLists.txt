cmake_minimum_required(VERSION 3.19)
project(KinectFusion-GPU CUDA C CXX)

set(CMAKE_CUDA_STANDARD 14)


find_package(Eigen3 REQUIRED)
find_package(CUDA 6.5 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

IF ((CUDA_VERSION VERSION_GREATER "10.0") OR (CUDA_VERSION VERSION_EQUAL "10.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
ENDIF()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(FreeImage_INCLUDE_DIR /usr/include/)
set(FreeImage_LIBRARY_DIR /usr/lib/x86_64-linux-gnu/)

set(HEADERS
        data_types.h
        Eigen.h
        common_functions.h
        FreeImageHelper.h
        VirtualSensor_freiburg.h
        surface_measurement.h)

set(SOURCES
        main.cu
        FreeImageHelper.cpp
        surface_measurement.cu)

add_executable(run-kf-gu ${SOURCES} ${HEADERS})
message("OpenCV_INCLUDE_DIRS" ${OpenCV_INCLUDE_DIRS})
message("OpenCV_LIBRARY_DIRS" "ASDASD" ${OpenCV_LIBS})
link_directories(${FreeImage_LIBRARY_DIR})
target_link_libraries(run-kf-gu Eigen3::Eigen freeimage ${OpenCV_LIBS})
target_include_directories(run-kf-gu PUBLIC ${OpenCV_INCLUDE_DIRS})

set_target_properties(
        run-kf-gu
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)