//
// Created by eralpkocas on 24.05.21.
//

#ifndef TUM_IN2354_KINECTFUSION_DATA_TYPES_H
#define TUM_IN2354_KINECTFUSION_DATA_TYPES_H

#include "Eigen.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include "opencv2/core/mat.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <iostream>


struct Pose
{
    Matrix4f m_trajectory;
    Matrix4f m_trajectoryInv;
    Pose(const Matrix4f& _m_trajectory, const Matrix4f& _m_trajectoryInv)
    {
        m_trajectory = _m_trajectory;
        m_trajectoryInv = _m_trajectoryInv;
    };
};

struct ImageConstants
{
    int num_levels = 3;
    float fX;
    float fY;
    float cX;
    float cY;
    Matrix4f m_trajectory;
    Matrix4f m_trajectoryInv;
    Matrix3f m_depthIntrinsics;
    Matrix3f m_depthIntrinsicsInv;
    Matrix4f m_depthExtrinsics;
    Matrix4f m_depthExtrinsicsInv;
    unsigned int m_colorImageWidth;
    unsigned int m_colorImageHeight;
    unsigned int m_depthImageWidth;
    unsigned int m_depthImageHeight;

    ImageConstants(float _fX, float _fY, float _cX, float _cY, const Matrix4f& _m_trajectory, const Matrix4f& _m_trajectoryInv,
                   const Matrix3f& _m_depthIntrinsics, const Matrix3f& _m_depthIntrinsicsInv, const Matrix4f& _m_depthExtrinsics,
                   const Matrix4f& _m_depthExtrinsicsInv, unsigned int _m_colorImageWidth,
                   unsigned int _m_colorImageHeight, unsigned int _m_depthImageWidth, unsigned int _m_depthImageHeight){
        fX = _fX;
        fY = _fY;
        cX = _cX;
        cY = _cY;
        m_trajectory << _m_trajectory;
        m_trajectoryInv << _m_trajectoryInv;
        m_depthIntrinsics << _m_depthIntrinsics;
        m_depthIntrinsicsInv << _m_depthIntrinsicsInv;
        m_depthExtrinsics << _m_depthExtrinsics;
        m_depthExtrinsicsInv << _m_depthExtrinsicsInv;
        m_colorImageWidth = _m_colorImageWidth;
        m_colorImageHeight = _m_colorImageHeight;
        m_depthImageWidth = _m_depthImageWidth;
        m_depthImageHeight = _m_depthImageHeight;
    };
};

struct ImageData
{
    cv::cuda::GpuMat m_depthMap;
    cv::cuda::GpuMat m_colorMap;

    ImageData(unsigned int _level_img_width, unsigned int _level_img_height,
              cv::Mat _m_depthMap, cv::Mat _m_colorMap){
        cv::cuda::createContinuous(_level_img_height, _level_img_width, CV_32F, m_depthMap);
        cv::cuda::createContinuous(_level_img_height, _level_img_width, CV_8UC4, m_colorMap);

        m_depthMap.upload(_m_depthMap);
        m_colorMap.upload(_m_colorMap);
    };
};

struct SurfaceLevelData
{
    std::vector<cv::cuda::GpuMat> curr_level_data;
    std::vector<cv::cuda::GpuMat> curr_smoothed_data;
    int level;
    std::vector<unsigned int> level_img_width;
    std::vector<unsigned int> level_img_height;
    std::vector<float> level_fX;
    std::vector<float> level_fY;
    std::vector<float> level_cX;
    std::vector<float> level_cY;
    std::vector<cv::cuda::GpuMat> vertex_map;
    std::vector<cv::cuda::GpuMat> normal_map;
    cv::cuda::GpuMat color_map;
    std::vector<cv::cuda::GpuMat> vertex_map_predicted;
    std::vector<cv::cuda::GpuMat> normal_map_predicted;
    std::vector<cv::cuda::GpuMat> color_map_predicted;
    //TODO: change color map to multilevel here since we will write color map multileve in surface prediction
    //TODO: we don't need image constants and image data structs anymore.

    SurfaceLevelData(int _level, unsigned int _level_img_width, unsigned int _level_img_height,
                     float _level_fX, float _level_fY, float _level_cX, float _level_cY, cv::cuda::GpuMat _color_map){
        level = _level;

        for(int i=0; i < level; i++){
            auto scale = (float) cv::pow(2, i);

            level_img_width.push_back(_level_img_width / (unsigned int) scale);
            level_img_height.push_back(_level_img_height / (unsigned int) scale);
            level_fX.push_back(_level_fX / scale);
            level_fY.push_back(_level_fY / scale);
            level_cX.push_back(_level_cX / scale);
            level_cY.push_back(_level_cY / scale);


            curr_level_data.push_back(cv::cuda::createContinuous(_level_img_height / scale, _level_img_width / scale, CV_32FC1));
            curr_smoothed_data.push_back(cv::cuda::createContinuous(_level_img_height / scale, _level_img_width / scale, CV_32FC1));
            vertex_map.push_back(cv::cuda::createContinuous(_level_img_height / scale, _level_img_width / scale, CV_32FC3));
            normal_map.push_back(cv::cuda::createContinuous(_level_img_height / scale, _level_img_width / scale, CV_32FC3));
            vertex_map_predicted.push_back(cv::cuda::createContinuous(_level_img_height / scale, _level_img_width / scale, CV_32FC3));
            normal_map_predicted.push_back(cv::cuda::createContinuous(_level_img_height / scale, _level_img_width / scale, CV_32FC3));
            color_map_predicted.push_back(cv::cuda::createContinuous(_level_img_height / scale, _level_img_width / scale, CV_8UC4));
            if(i==0){
                cv::cuda::createContinuous(_level_img_height / scale, _level_img_width / scale, CV_8UC4, color_map);
                color_map = _color_map;
            }
        }
    }

};

struct GlobalVolume
{
    // position stored as 4 floats (4th component is supposed to be 1.0)
    cv::cuda::GpuMat TSDF_values;
    cv::cuda::GpuMat TSDF_weight;
    // color stored as 4 unsigned char
    cv::cuda::GpuMat TSDF_color;
    Vector3f min;
    Vector3f max;
    float voxel_scale;
    int3 volume_size;
    float truncation_distance;

    GlobalVolume(const int3 _volume_size, const float _voxel_scale, float _truncation_distance,Vector3f _min, Vector3f _max){
        cv::cuda::createContinuous(_volume_size.x * _volume_size.y, _volume_size.z, CV_32F, TSDF_values);
        cv::cuda::createContinuous(_volume_size.x * _volume_size.y, _volume_size.z, CV_8UC1, TSDF_weight);
        cv::cuda::createContinuous(_volume_size.x * _volume_size.y, _volume_size.z, CV_8UC4, TSDF_color);
        TSDF_values.setTo(0);
        TSDF_weight.setTo(0);
        TSDF_color.setTo(0);

        volume_size = _volume_size;
        voxel_scale = _voxel_scale;
        truncation_distance = _truncation_distance;
        min = _min;
        max = _max;
    }

    int getDimX(){
        return volume_size.x;
    }
    int getDimY(){
        return volume_size.y;
    }
    int getDimZ(){
        return volume_size.z;
    }

    //! Returns the corresponding node of given cartesian coordinates.
    inline Vector3f compute_grid(Vector3f p)
    {
        return Vector3f(((p[0] - min[0]) / (max[0] - min[0])) / volume_size.x,
                        ((p[1] - min[1]) / (max[1] - min[1])) / volume_size.y,
                        ((p[2] - min[2]) / (max[2] - min[2])) / volume_size.z
        );
    }
};


//struct CameraRefPoints
//{
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//    float curr_fX, curr_fY, curr_cX, curr_cY;
//    float img_width;
//    float img_height;
//
//    // position stored as 4 floats (4th component is supposed to be 1.0)
//    Vector3f position;
//    // color stored as 4 unsigned char
//    Vector4uc color;
//
//    CameraRefPoints getCurrLevel(int level){
//        if(level == 0) return *this;
//        auto scale = (float) ceres::pow(2, i);
//        return CameraRefPoints { curr_fX / scale, curr_fY / scale, curr_cX / scale, curr_cY / scale,
//                          img_width / scale, img_height / scale,
//                          Vector3f(0,0,0),Vector4uc(0,0,0,0)
//        };
//    }
//};
//
//struct GlobalPoints
//{
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//    // position stored as 4 floats (4th component is supposed to be 1.0)
//    Vector4f position;
//    // color stored as 4 unsigned char
//    Vector4uc color;
//};
//
//struct SurfaceLevelData
//{
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//
//            cv::Mat curr_level_data;
//    cv::Mat curr_smoothed_data;
//
//    std::vector<Vector3f> vertex_map;
//    std::vector<Vector3f> normal_map;
//    std::vector<Vector3f> vertex_map_predicted;
//    std::vector<Vector3f> normal_map_predicted;
//
//};
//
//struct ImageProperties{
//    int num_levels = 3;
//    float fX;
//    float fY;
//    float cX;
//    float cY;
//    float truncation_distance;
//
//    cv::Mat m_depthMap;
//    BYTE *m_colorMap;
//    //cv::Mat m_colorMap; // TODO: it is wrong!!! check initialization and type. Correct color update in SurfaceReconstructionUpdate.
//    Matrix4f m_trajectory;
//    Matrix4f m_trajectoryInv;
//    Matrix3f m_depthIntrinsics;
//    Matrix4f m_depthExtrinsics;
//    unsigned int m_colorImageWidth;
//    unsigned int m_colorImageHeight;
//    unsigned int m_depthImageWidth;
//    unsigned int m_depthImageHeight;
//    CameraRefPoints **camera_reference_points;
//    GlobalPoints **global_points;
//    SurfaceLevelData *all_data;
//    //Volume* global_tsdf;
//};

#endif //TUM_IN2354_KINECTFUSION_DATA_TYPES_H
