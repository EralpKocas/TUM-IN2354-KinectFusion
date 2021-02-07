#pragma once

#ifndef KINECTFUSION_COMMON_H
#define KINECTFUSION_COMMON_H

#include <array>
#include "Eigen.h"
#include "ceres/ceres.h"
#include "Volume.h"
#include "VirtualSensor_freiburg.h"
//#include "opencv2/opencv.hpp"
#include <opencv2/core/mat.hpp>
#include "opencv2/imgproc/imgproc.hpp"

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

    cv::Mat curr_level_data;
    cv::Mat curr_smoothed_data;
    float img_width;
    float img_height;
    float curr_fX;
    float curr_fY;
    float curr_cX;
    float curr_cY;
    std::vector<Vector3f> vertex_map;
    std::vector<Vector3f> normal_map;
    std::vector<Vector3f> vertex_map_predicted;
    std::vector<Vector3f> normal_map_predicted;

};

struct ImageProperties{
    int num_levels = 3;
    float fX;
    float fY;
    float cX;
    float cY;
    float truncation_distance;

    cv::Mat m_depthMap;
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
    Volume* global_tsdf;
};

// compute 3d camera reference points
void compute_camera_ref_points(ImageProperties* imageProperties, int level)
{
    int numWH = imageProperties->all_data[level].img_width * imageProperties->all_data[level].img_height;
    imageProperties->camera_reference_points = new CameraRefPoints[numWH];
    for(int i=0; i < numWH; i++){
        if(imageProperties->m_depthMap.at<float>(i) == MINF){
            imageProperties->camera_reference_points[i].position = Vector3f(MINF, MINF, MINF);
            imageProperties->global_points[i].color = Vector4uc(0,0,0,0);
        }
        else{
            int pixel_y = i / imageProperties->m_depthImageWidth;
            int pixel_x = i - pixel_y * imageProperties->m_depthImageWidth;
            float currDepthValue = imageProperties->m_depthMap.at<float>(i);
            float camera_x = currDepthValue * ((float) pixel_x - imageProperties->cX) / imageProperties->fX;
            float camera_y = currDepthValue * ((float) pixel_y - imageProperties->cY) / imageProperties->fY;

            imageProperties->camera_reference_points[i].position = Vector3f(camera_x, camera_y, currDepthValue);
            imageProperties->camera_reference_points[i].color = Vector4uc(imageProperties->m_colorMap[4*i], imageProperties->m_colorMap[4*i+1], imageProperties->m_colorMap[4*i+2], imageProperties->m_colorMap[4*i+3]);
        }
    }
}

// compute global 3D points
void compute_global_points(ImageProperties* imageProperties, int level)
{
    int numWH = imageProperties->all_data[level].img_width * imageProperties->all_data[level].img_height;
    imageProperties->global_points = new GlobalPoints[numWH];

    for(int i=0; i < numWH; i++) {
        if (imageProperties->m_depthMap.at<float>(i) == MINF) {
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
}

Vector3f get_translation(ImageProperties* image_properties){
    return image_properties->m_trajectory.block<3, 1>(0, 3);
}

Matrix3f get_rotation(ImageProperties* image_properties){
    return image_properties->m_trajectory.block<3, 3>(0, 0);
}


#endif //KINECTFUSION_COMMON_H
