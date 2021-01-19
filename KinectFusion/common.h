
#include "VirtualSensor_freiburg.h"


#ifndef KINECTFUSION_COMMON_H
#define KINECTFUSION_COMMON_H

struct CameraRefPoints
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // position stored as 4 floats (4th component is supposed to be 1.0)
    Vector3f position;
    // color stored as 4 unsigned char
    Vector4uc color;
};

struct GlobalPoints
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4f position;
    // color stored as 4 unsigned char
    Vector4uc color;
};

struct SurfaceLevelData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // will create a data structure for holding all data for all levels!
    float* curr_level_data;
    float* curr_smoothed_data;
    BYTE* curr_level_color;
    float curr_fX;
    float curr_fY;
    float curr_cX;
    float curr_cY;

};


struct ImageProperties{
    float fX;
    float fY;
    float cX;
    float cY;

    float *m_depthMap;
    BYTE *m_colorMap;
    Matrix4f m_trajectory;
    Matrix4f m_trajectoryInv;
    Matrix3f m_depthIntrinsics;
    Matrix4f m_depthExtrinsics;
    unsigned int m_colorImageWidth;
    unsigned int m_colorImageHeight;
    unsigned int m_depthImageWidth;
    unsigned int m_depthImageHeight;
    CameraRefPoints *camera_reference_points;
    GlobalPoints *global_points;
    SurfaceLevelData *all_data;
};


    // compute 3d camera reference points
    void compute_camera_ref_points(ImageProperties* imageProperties)
    {
        int numWH = imageProperties->m_depthImageWidth * imageProperties->m_depthImageHeight;

        for(int i=0; i < numWH; i++){
            if(imageProperties->m_depthMap[i] == MINF){
                imageProperties->camera_reference_points[i].position = Vector3f(MINF, MINF, MINF);
                imageProperties->global_points[i].color = Vector4uc(0,0,0,0);
            }
            else{
                int pixel_y = i / imageProperties->m_depthImageWidth;
                int pixel_x = i - pixel_y * imageProperties->m_depthImageWidth;
                float currDepthValue = imageProperties->m_depthMap[i];
                float camera_x = currDepthValue * ((float) pixel_x - imageProperties->cX) / imageProperties->fX;
                float camera_y = currDepthValue * ((float) pixel_y - imageProperties->cY) / imageProperties->fY;

                imageProperties->camera_reference_points[i].position = Vector3f(camera_x, camera_y, currDepthValue);
                imageProperties->camera_reference_points[i].color = Vector4uc(imageProperties->m_colorMap[4*i], imageProperties->m_colorMap[4*i+1], imageProperties->m_colorMap[4*i+2], imageProperties->m_colorMap[4*i+3]);
            }
        }
        for(int i=0; i < numWH; i++) {
            if (imageProperties->m_depthMap[i] != MINF) {
                std::cout << "first camera ref point at i: \n" << i << std::endl;
                std::cout << "curr depth: \n" << imageProperties->m_depthMap[i] << std::endl;
                std::cout << "camera_reference_points: \n" << imageProperties->camera_reference_points[i].position << std::endl;
                break;
            }
        }
    }

    // compute global 3D points
    void compute_global_points(ImageProperties* imageProperties)
    {
        int numWH = imageProperties->m_depthImageWidth * imageProperties->m_depthImageHeight;

        for(int i=0; i < numWH; i++) {
            if (imageProperties->m_depthMap[i] == MINF) {
                imageProperties->global_points[i].position = Vector4f(MINF, MINF, MINF, MINF);
                imageProperties->global_points[i].color = Vector4uc(0, 0, 0, 0);
            } else {
                Vector4f camera_ref_vector = Vector4f(imageProperties->camera_reference_points[i].position.x(),
                                                      imageProperties->camera_reference_points[i].position.y(),
                                                      imageProperties->camera_reference_points[i].position.z(),
                                                      (float) 1.0);

                Vector4f global_point = imageProperties->m_trajectory * camera_ref_vector;

                imageProperties->global_points[i].position = global_point;
                imageProperties->global_points[i].color = imageProperties->camera_reference_points[i].color;
            }
        }
        for(int i=0; i < numWH; i++) {
            if (imageProperties->m_depthMap[i] != MINF) {
                std::cout << "first global point at i: \n" << i << std::endl;
                std::cout << "m_trajectory: \n" << imageProperties->m_trajectory << std::endl;
                std::cout << "global: \n" << imageProperties->global_points[i].position << std::endl;
                break;
            }
        }
    }


#endif //KINECTFUSION_COMMON_H
