#include <iostream>
//#include <fstream>
#include <array>

#include <common.h>
#include "VirtualSensor_freiburg.h"

#include "SurfaceMeasurement.h"
#include "SurfaceReconstructionUpdate.h"
//#include "SurfacePrediction.h"

ImageProperties* init(VirtualSensor_freiburg &sensor)
{
    ImageProperties* imageProperties = new ImageProperties();

    imageProperties->m_depthMap = cv::Mat(640*480, 1, CV_32F, sensor.getDepth());
    imageProperties->m_colorMap = sensor.getColorRGBX();
    imageProperties->m_trajectory = sensor.getTrajectory();
    imageProperties->m_trajectoryInv = sensor.getTrajectory().inverse();
    imageProperties->m_depthIntrinsics = sensor.getDepthIntrinsics();
    imageProperties->m_depthExtrinsics = sensor.getDepthExtrinsics();

    imageProperties->fX = sensor.getDepthIntrinsics().coeffRef(0,0);
    imageProperties->fY = sensor.getDepthIntrinsics().coeffRef(1,1);
    imageProperties->cX = sensor.getDepthIntrinsics().coeffRef(0,2);
    imageProperties->cY = sensor.getDepthIntrinsics().coeffRef(1,2);

    imageProperties->m_colorImageWidth = 640;
    imageProperties->m_colorImageHeight = 480;
    imageProperties->m_depthImageWidth = 640;
    imageProperties->m_depthImageHeight = 480;

    imageProperties->camera_reference_points = new CameraRefPoints[imageProperties->m_depthImageWidth * imageProperties->m_depthImageHeight];
    imageProperties->global_points = new GlobalPoints[imageProperties->m_depthImageWidth * imageProperties->m_depthImageHeight];
    return imageProperties;
}


int main() {

    // Make sure this path points to the data folder
    std::string filenameIn = "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/data/rgbd_dataset_freiburg1_xyz/";
    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor_freiburg sensor;
    if (!sensor.init(filenameIn))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    // convert video to meshes
    while (sensor.processNextFrame()) {
        ImageProperties* imageProperties = init(sensor);
        // get ptr to the current depth frame
        // depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())

        //float *depthMap = imageProperties->m_depthMap;

        // get ptr to the current color frame
        // color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
        BYTE *colorMap = imageProperties->m_colorMap;

        // get depth intrinsics
        Matrix3f depthIntrinsics = imageProperties->m_depthIntrinsics;

        // compute inverse depth extrinsics
        Matrix4f depthExtrinsicsInv = (imageProperties->m_depthExtrinsics).inverse();

        Matrix4f trajectory = imageProperties->m_trajectory;
        Matrix4f trajectoryInv = imageProperties->m_trajectoryInv;

        SurfaceMeasurement surface_measurement;
        surface_measurement.surface_measurement_pipeline(imageProperties);
        /*if(!surface_measurement.init(depthMap, colorMap, trajectory, trajectoryInv, depthIntrinsics))
        {
            std::cout << "Failed to read and assign data!" << std::endl;
            return -1;
        }*/

    }

        return 0;
}

