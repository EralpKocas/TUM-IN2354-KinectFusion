#include <iostream>
#include <iostream>
#include <fstream>
#include <string>

#include <math.h>
#include "common_functions.h"
#include "data_types.h"
#include "VirtualSensor_freiburg.h"
//#include <opencv2/core/mat.hpp>
//#include "opencv2/imgproc/imgproc.hpp"
//#include <opencv2/highgui.hpp>
#include "surface_measurement.h"
#include "SimpleMesh.h"
#include "pose_estimation.h"

int main() {
    //std::cout << "Hello, World!" << std::endl; std::string filenameIn = "/home/ilteber/data/rgbd_dataset_freiburg1_xyz/";
    std::string filenameIn = "/media/eralpkocas/hdd/TUM/3D_Scanning/data/rgbd_dataset_freiburg1_xyz/";
    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    bool isFirstFrame = true;
    VirtualSensor_freiburg sensor;
    if (!sensor.init(filenameIn))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }
    std::vector<int> iterations = {4, 5, 10};

    Pose pose_struct = {
            Matrix4f::Zero(4, 4),
            Matrix4f::Zero(4, 4),
    };

    int i = 0;
    while (sensor.processNextFrame()) {

        ImageConstants img_constants = {
                sensor.getDepthIntrinsics().coeffRef(0, 0),
                sensor.getDepthIntrinsics().coeffRef(1,1),
                sensor.getDepthIntrinsics().coeffRef(0,2),
                sensor.getDepthIntrinsics().coeffRef(1,2),
                sensor.getTrajectory(),
                sensor.getTrajectory().inverse(),
                sensor.getDepthIntrinsics(),
                sensor.getDepthExtrinsics(),
                sensor.getDepthExtrinsics().inverse(),
                sensor.getColorImageWidth(),
                sensor.getColorImageHeight(),
                sensor.getDepthImageWidth(),
                sensor.getDepthImageHeight(),
        };

        ImageData img_data = {
                sensor.getDepthImageWidth(),
                sensor.getDepthImageHeight(),
                cv::Mat(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32F, sensor.getDepth()),
                cv::Mat(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_8UC4, sensor.getColorRGBX()),
        };

        if(i==0){
            pose_struct.m_trajectory = img_constants.m_trajectory;
            pose_struct.m_trajectoryInv = img_constants.m_trajectoryInv;
        }
        // TODO: inverse trajectory is nan for all indices. check!
        SurfaceLevelData surf_data = {
                3,
                img_constants.m_colorImageWidth,
                img_constants.m_colorImageHeight,
                img_constants.fX,
                img_constants.fY,
                img_constants.cX,
                img_constants.cY,
        };
//        cv::Mat result;
//        img_data.m_depthMap.download(result);
//        cv::imshow("result", result);
//        cv::waitKey(30);

        //std::cout << "line 50: "  << result << std::endl;
        //std::cout << "line 51: "  << img_data.m_colorMap << std::endl;

        // step 1: Surface Measurement
        surface_measurement_pipeline(&surf_data, img_data, img_constants);

        // step 2: Pose Estimation, for frame == 0, don't perform
        if(!isFirstFrame){
            pose_estimate(iterations, &img_constants, &img_data, &surf_data, &pose_struct);
            std::cout << "frame: " << i << std::endl;
            std::cout << "rotation: " << img_constants.m_trajectory.block<3, 3>(0, 0) << std::endl;
            std::cout << "translation: " << img_constants.m_trajectory.block<3, 1>(0, 3) << std::endl;
        }else{
            isFirstFrame = false;
        }
        // step 3: Surface Reconstruction Update
        // step 4: Raycast Prediction

        SimpleMesh mesh;
        std::stringstream ss;
        i++;
        //ss << "result_" << i++ << ".off";
        //cv::Mat result;
        //surf_data.vertex_map[0].download(result);

        //cv::Mat color_map;
        //img_data.m_colorMap.download(color_map);
//        if (!mesh.WriteMesh2(result, color_map, surf_data.level_img_width[0],
//                             surf_data.level_img_height[0], ss.str()))
//        {
//            std::cout << "ERROR: unable to write output file!" << std::endl;
//            return -1;
//        }
    }
    return 0;
}
