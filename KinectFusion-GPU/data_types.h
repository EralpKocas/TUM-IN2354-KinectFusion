//
// Created by eralpkocas on 24.05.21.
//

#ifndef TUM_IN2354_KINECTFUSION_DATA_TYPES_H
#define TUM_IN2354_KINECTFUSION_DATA_TYPES_H

/*
struct CameraRefPoints
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    float curr_fX, curr_fY, curr_cX, curr_cY;
    float img_width;
    float img_height;

    // position stored as 4 floats (4th component is supposed to be 1.0)
    Vector3f position;
    // color stored as 4 unsigned char
    Vector4uc color;

    CameraRefPoints getCurrLevel(int level){
        if(level == 0) return *this;
        auto scale = (float) ceres::pow(2, i);
        return CameraRefPoints { curr_fX / scale, curr_fY / scale, curr_cX / scale, curr_cY / scale,
                          img_width / scale, img_height / scale,
                          Vector3f(0,0,0),Vector4uc(0,0,0,0)
        };
    }
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
    //cv::Mat m_colorMap; // TODO: it is wrong!!! check initialization and type. Correct color update in SurfaceReconstructionUpdate.
    Matrix4f m_trajectory;
    Matrix4f m_trajectoryInv;
    Matrix3f m_depthIntrinsics;
    Matrix4f m_depthExtrinsics;
    unsigned int m_colorImageWidth;
    unsigned int m_colorImageHeight;
    unsigned int m_depthImageWidth;
    unsigned int m_depthImageHeight;
    CameraRefPoints **camera_reference_points;
    GlobalPoints **global_points;
    SurfaceLevelData *all_data;
    //Volume* global_tsdf;
};
*/
#endif //TUM_IN2354_KINECTFUSION_DATA_TYPES_H
