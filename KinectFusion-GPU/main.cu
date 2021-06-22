#include <iostream>
#include <fstream>
#include <string>

#include <math.h>
#include "common_functions.h"
#include "data_types.h"
#include "VirtualSensor_freiburg.h"
#include <opencv2/core/mat.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>

int main(void)
    {


    // Make sure this path points to the data folder
    //std::string filenameIn = "/Users/beyzatugcebilgic/Desktop/TUM-IN2354-KinectFusion/KinectFusion/data/rgbd_dataset_freiburg1_xyz/";
    std::string filenameIn = "/media/eralpkocas/hdd/TUM/3D_Scanning/data/rgbd_dataset_freiburg1_xyz/";

//  std::string filenameIn = "/home/ilteber/data/rgbd_dataset_freiburg1_xyz/";
    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    bool isFirstFrame = true;
    VirtualSensor_freiburg sensor;
    if (!sensor.init(filenameIn))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    ImageConstants img_constants = {
            sensor.getDepthIntrinsics().coeffRef(0, 0),
            sensor.getDepthIntrinsics().coeffRef(1,1),
            sensor.getDepthIntrinsics().coeffRef(0,2),
            sensor.getDepthIntrinsics().coeffRef(1,2),
            sensor.getTrajectory(),
            sensor.getTrajectory().inverse(),
            sensor.getDepthIntrinsics(),
            sensor.getDepthExtrinsics(),
            sensor.getColorImageWidth(),
            sensor.getColorImageHeight(),
            sensor.getDepthImageWidth(),
            sensor.getDepthImageHeight(),
    };

    while (sensor.processNextFrame()) {
        ImageData img_data = {
                sensor.getDepthImageWidth(),
                sensor.getDepthImageHeight(),
                cv::Mat(sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), CV_32F, sensor.getDepth()),
                cv::Mat(sensor.getColorImageWidth(), sensor.getColorImageHeight(), CV_8U, sensor.getColorRGBX()),
                //cv::cuda::GpuMat(640, 480, CV_32F, sensor.getDepth()),
                //cv::cuda::GpuMat(640, 480, CV_8U, sensor.getColorRGBX()),
        };
        // TODO: inverse trajectory is nan for all indices. check!

        cv::Mat result;
        img_data.m_depthMap.download(result);
        cv::imshow("result", result);
        //std::cout << "line 50: "  << result << std::endl;
        //std::cout << "line 51: "  << img_data.m_colorMap << std::endl;
        break;
        // step 1: Surface Measurement
        // step 2: Pose Estimation, for frame == 0, don't perform
        if(!isFirstFrame){

        }else{
            isFirstFrame = false;
        }
        // step 3: Surface Reconstruction Update
        // step 4: Raycast Prediction
    }

    return 0;
    }