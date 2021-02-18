#include <iostream>
//#include <fstream>
#include <array>

#include <iostream>
#include <fstream>
#include <string>

#include <common.h>
#include "VirtualSensor_freiburg.h"

#include "SurfaceMeasurement.h"
#include "SurfaceReconstructionUpdate.h"
#include "PoseEstimation.h"
#include "SurfacePrediction.h"
#include "SimpleMesh.h"
#include "MarchingCubes.h"

#define MIN_POINT -1.5f, -1.0f, -0.1f
#define MAX_POINT 1.5f, 1.0f, 3.5f
#define RESOLUTION 20, 20, 20

ImageProperties* init(VirtualSensor_freiburg &sensor)
{
    ImageProperties* imageProperties = new ImageProperties();

    imageProperties->m_depthMap = cv::Mat(640, 480, CV_32F, sensor.getDepth());
    imageProperties->m_colorMap = sensor.getColorRGBX();
    //imageProperties->m_colorMap = cv::Mat(640*480, 1, CV_8UC4, sensor.getColorRGBX());
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

    imageProperties->truncation_distance = 25.f; // find the correct value!

    //imageProperties->camera_reference_points = new CameraRefPoints*[imageProperties->num_levels];

    //imageProperties->camera_reference_points = new CameraRefPoints[imageProperties->m_depthImageWidth * imageProperties->m_depthImageHeight];

    //imageProperties->global_points = new GlobalPoints*[imageProperties->num_levels];

    //imageProperties->global_points = new GlobalPoints[imageProperties->m_depthImageWidth * imageProperties->m_depthImageHeight];
    //imageProperties->global_tsdf = new Volume(Vector3f(MIN_POINT), Vector3f(MAX_POINT), RESOLUTION, 3);
    return imageProperties;
}

void initSurfaceLevelData(ImageProperties* imageProperties){
    imageProperties->all_data = new SurfaceLevelData[imageProperties->num_levels];
    for(int i = 0; i < imageProperties->num_levels; i++){
        imageProperties->all_data[i].img_width = imageProperties->m_depthImageWidth;
        imageProperties->all_data[i].img_height = imageProperties->m_depthImageHeight;
    }
}


int main() {

    // Make sure this path points to the data folder
    //std::string filenameIn = "/Users/beyzatugcebilgic/Desktop/TUM-IN2354-KinectFusion/KinectFusion/data/rgbd_dataset_freiburg1_xyz/";
    std::string filenameIn = "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/data/rgbd_dataset_freiburg1_xyz/";
    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor_freiburg sensor;
    if (!sensor.init(filenameIn))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    Volume* global_volume = new Volume(Vector3f(MIN_POINT), Vector3f(MAX_POINT), RESOLUTION, 3);

    // convert video to meshes
    int i = 1;
    while (sensor.processNextFrame()) {
        ImageProperties* imageProperties = init(sensor);
        initSurfaceLevelData(imageProperties);

        //compute_camera_ref_points(imageProperties);
        //compute_global_points(imageProperties);

        // get ptr to the current depth frame
        // depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())

        //float *depthMap = imageProperties->m_depthMap;

        // get ptr to the current color frame
        // color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
        //BYTE *colorMap = imageProperties->m_colorMap;

        std::vector<int> iterations = {4, 5, 10};

        //std::ofstream out("out.txt");
        //std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        //std::cout.rdbuf(out.rdbuf());

        SurfaceMeasurement surface_measurement;
        surface_measurement.surface_measurement_pipeline(imageProperties);

        if ( i != 1){
            PoseEstimation pose_estimation;
            pose_estimation.estimate_pose(iterations, imageProperties);
        }

        SurfaceReconstructionUpdate reconstruction_update;
        reconstruction_update.updateSurfaceReconstruction(imageProperties, global_volume);

        SurfacePrediction surface_prediction;
        surface_prediction.predict_surface(imageProperties, global_volume);

        /*if(!surface_measurement.init(depthMap, colorMap, trajectory, trajectoryInv, depthIntrinsics))
        {
            std::cout << "Failed to read and assign data!" << std::endl;
            return -1;
        }*/
        // extract the zero iso-surface using marching cubes
        SimpleMesh mesh;
        for (unsigned int x = 0; x < global_volume->getDimX() - 1; x++)
        {
            for (unsigned int y = 0; y < global_volume->getDimY() - 1; y++)
            {
                for (unsigned int z = 0; z < global_volume->getDimZ() - 1; z++)
                {
                    //if (imageProperties->global_tsdf->get(x, y, z).is_occupied){
                    ProcessVolumeCell(global_volume, x, y, z, (double)0.00, &mesh);
                    //}
                }
            }
        }

        // write mesh to file
        std::stringstream ss;

        ss << "result_" << i++ << ".off";
        if (!mesh.WriteMesh(ss.str()))
        {
            std::cout << "ERROR: unable to write output file!" << std::endl;
            return -1;
        }
        //delete imageProperties;
    }

    return 0;
}

