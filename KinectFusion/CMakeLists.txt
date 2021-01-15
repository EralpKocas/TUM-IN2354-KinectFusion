cmake_minimum_required(VERSION 3.16)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(KinectFusion)
include_directories("/usr/local/include")
include_directories("/usr/local/Cellar")

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs CACHE PATH "Path to lib folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/Eigen/ CACHE PATH "Path to Eigen source folder")
set(glog_INCLUDE_DIR ${LIBRARY_DIR}/glog-lib/include CACHE PATH "Path to glog source folder")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to Ceres build folder")
set(Ceres_INCLUDE_DIR ${Ceres_DIR}/../include)
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann/ CACHE PATH "Path to Flann source folder")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Ceres_DIR})

set(FreeImage_INCLUDE_DIR /usr/local/include)
set(FreeImage_LIBRARY_DIR /usr/local/Cellar/freeimage/3.18.0/lib/)

set(CMAKE_CXX_STANDARD 14)

find_package(Ceres REQUIRED)
find_package(glog REQUIRED ${glog_DIR_HINT})

message(STATUS "ceres library name: ${Ceres_LIBRARIES}")
message(STATUS "path to ceres cmake information: ${Ceres_DIR}")
message(STATUS "path to ceres include directory: ${Ceres_INCLUDE_DIR}")
message(STATUS "glog library name: ${glog_LIBRARIES}")
message(STATUS "path to glog include directory = ${glog_INCLUDE_DIR}")

set(HEADER_FILES
        Eigen.h
        VirtualSensor_freiburg.h
        VirtualSensor_office.h
        NearestNeighbor.h
        FreeImageHelper.h
        SurfaceMeasurement.h
        )
set(SOURCE_FILES
        FreeImageHelper.cpp
        )

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(KinectFusion main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(KinectFusion ceres freeimage)
target_include_directories(KinectFusion PUBLIC ${Eigen3_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}  ${Ceres_INCLUDE_DIR} ${glog_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR})
