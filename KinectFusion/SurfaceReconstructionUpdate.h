#include <array>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "Eigen.h"
#include "ceres/ceres.h"
#include "VirtualSensor_freiburg.h"



class SurfaceReconstructionUpdate{

public:
    SurfaceReconstructionUpdate() {}
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

    bool init(float* depthMap, BYTE*colorMap, Matrix4f& trajectory,
              Matrix4f& trajectoryInv, Matrix3f& depthIntrinsics)
    {
        m_depthMap = depthMap;
        m_colorMap = colorMap;
        m_trajectory = trajectory;
        m_trajectoryInv = trajectoryInv;
        m_depthIntrinsics = depthIntrinsics;

        fX = m_depthIntrinsics(0, 0);
        fY = m_depthIntrinsics(1, 1);
        cX = m_depthIntrinsics(0, 2);
        cY = m_depthIntrinsics(1, 2);

        num_levels = 3;
        bilateral_color_sigma = 1.0f;
        bilateral_spatial_sigma = 1.0f;
        depth_diameter = 3 * (int) bilateral_spatial_sigma;

        m_colorImageWidth = 640;
        m_colorImageHeight = 480;
        m_depthImageWidth = 640;
        m_depthImageHeight = 480;

        camera_reference_points = new CameraRefPoints[m_depthImageWidth * m_depthImageHeight];
        global_points = new GlobalPoints[m_depthImageWidth * m_depthImageHeight];

        return true;
    }

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

    bool init_pyramid()
    {
        // TODO: will initialize pyramid for given number of layers
        // TODO: need to change camera parameters with scaling level and need to add cv::pyrDown() so that values are resized accordingly.
        all_data = new SurfaceLevelData[num_levels];
        for(int i=0; i < num_levels; i++)
        {
            all_data[i].curr_level_data = m_depthMap;
            all_data[i].curr_level_color = m_colorMap;
            all_data[i].curr_fX = fX;
            all_data[i].curr_fY = fY;
            all_data[i].curr_cX = cX;
            all_data[i].curr_cY = cY;
        }
        return true;
    }

    Vector4f convert_homogeneous_vector(Vector3f vector_3d)
    {
        return Vector4f(vector_3d.x(), vector_3d.y(), vector_3d.z(), (float) 1.0);
    }

    // compute 3d camera reference points
    void compute_camera_ref_points() {
        int numWH = m_depthImageWidth * m_depthImageHeight;

        for (int i = 0; i < numWH; i++) {
            if (m_depthMap[i] == MINF) {
                camera_reference_points[i].position = Vector3f(MINF, MINF, MINF);
                global_points[i].color = Vector4uc(0, 0, 0, 0);
            } else {
                int pixel_y = i / m_depthImageWidth;
                int pixel_x = i - pixel_y * m_depthImageWidth;
                float currDepthValue = m_depthMap[i];
                float camera_x = currDepthValue * ((float) pixel_x - cX) / fX;
                float camera_y = currDepthValue * ((float) pixel_y - cY) / fY;

                camera_reference_points[i].position = Vector3f(camera_x, camera_y, currDepthValue);
                camera_reference_points[i].color = Vector4uc(m_colorMap[4 * i], m_colorMap[4 * i + 1],
                                                             m_colorMap[4 * i + 2], m_colorMap[4 * i + 3]);
            }
        }
    }

    // TODO: calculate truncation function
    float* calculateSDF_truncation(float truncation_distance, float sdf){
        if (sdf >= -truncation_distance) {
            float new_tsdf = fmin(1.f, sdf / truncation_distance);
        }

    }

    // TODO: calculate current TSDF (FRk←calculatecurrenttsdf(Ψ,Rk,K,tg,k,p))
    //λ = ||K^-1*x||2

    float* calculateCurrentTSDF( float* depthMap, Matrix3f intrinsics, Vector3f camera_ref, int k, Vector3f p){
        Vector3f camera_pos = camera_ref - p;
        float current_tsdf = (1.f / calculateLambda(intrinsics, p)) * camera_pos.norm() - depthMap[k];
        return calculateSDF_truncation(truncation_distance, current_tsdf);
    }


    // TODO: calculate weighted running tsdf average
    float calculateWeightedTSDF(int current_weight, float current_tsdf, int new_weight, float new_tsdf){
        float updated_tsdf = (current_weight * current_tsdf + new_weight * new_tsdf) /
                             (current_weight + new_weight);
    }

    // TODO: calculate weighted running weight average
    float calculateWeightedAvgWeight(int current_weight, int new_weight){
        return current_weight + new_weight;
    }

    // TODO: truncate updated weight
    int calculateTruncatedWeight(int current_weight, int new_weight, int some_value){
        return std::min(current_weight + new_weight, some_value);
    }


    //HELPER FUNCTIONS
    float calculateLambda( Matrix3f intrinsics, Vector3f p){
        Vector2f projected = perspective_projection(p);
        Vector3f dot_p = Vector3f(projected.x(), projected.y(), 1.0f);
        return (intrinsics.inverse() * dot_p).norm();
    }

    Vector2f perspective_projection(Vector3f p)
    {
        Vector3f p2 = m_depthIntrinsics * m_depthExtrinsics.inverse() * p;
        return Vector2f(p2.x() / p2.z(), p2.y() / p2.z());
    }

private:

    float depth_margin;                 //μ
    float truncation_distance;          //η
    float *m_depthMap;                  //Rk

    Matrix3f m_depthIntrinsics;         //camera calibration matrix K
    Matrix4f m_depthExtrinsics;

        int num_levels;
        float bilateral_color_sigma;
        float bilateral_spatial_sigma;
        int depth_diameter;

        float fX;
        float fY;
        float cX;
        float cY;

        BYTE *m_colorMap;
        Matrix4f m_trajectory;
        Matrix4f m_trajectoryInv;
        unsigned int m_colorImageWidth;
        unsigned int m_colorImageHeight;
        unsigned int m_depthImageWidth;
        unsigned int m_depthImageHeight;
        CameraRefPoints *camera_reference_points;
        GlobalPoints *global_points;
        SurfaceLevelData *all_data;
};








