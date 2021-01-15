#include <iostream>
#include <fstream>
#include <array>

#include "Eigen.h"

#include "VirtualSensor_freiburg.h"
#include "VirtualSensor_office.h"
#include "SurfaceMeasurement.h"

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

        // get ptr to the current depth frame
        // depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
        float *depthMap = sensor.getDepth();

        // get ptr to the current color frame
        // color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
        BYTE *colorMap = sensor.getColorRGBX();

        // get depth intrinsics
        Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();
        float fX = depthIntrinsics(0, 0);
        float fY = depthIntrinsics(1, 1);
        float cX = depthIntrinsics(0, 2);
        float cY = depthIntrinsics(1, 2);

        // compute inverse depth extrinsics
        Matrix4f depthExtrinsicsInv = sensor.getDepthExtrinsics().inverse();

        Matrix4f trajectory = sensor.getTrajectory();
        Matrix4f trajectoryInv = sensor.getTrajectory().inverse();

        SurfaceMeasurement surface_measurement;
        if(!surface_measurement.init(depthMap, colorMap, trajectory, trajectoryInv, depthIntrinsics))
        {
            std::cout << "Failed to read and assign data!" << std::endl;
            return -1;
        }

    }

        return 0;
}
