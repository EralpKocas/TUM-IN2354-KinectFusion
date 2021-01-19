#include <array>

#include <opencv2/opencv.hpp>
#include "Eigen.h"
#include "ceres/ceres.h"
#include "common.h"


class SurfaceMeasurement {

public:

    // initialize
    //SurfaceMeasurement() : { }


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

        //cv::pyrDown(all_data*, , , cv::BORDER_DEFAULT);
        return true;
    }

    Vector4f convert_homogeneous_vector(Vector3f vector_3d)
    {
        return Vector4f(vector_3d.x(), vector_3d.y(), vector_3d.z(), (float) 1.0);
    }



    // compute bilateral filter
    // TODO: does not work unless we change all_data.curr_level_data to a proper array.
    void compute_bilateral_filter()
    {
        /*for(int i=0; i<num_levels;i++){
            cv::bilateralFilter(all_data[i].curr_level_data, all_data[i].curr_smoothed_data,
                    depth_diameter, bilateral_color_sigma, bilateral_spatial_sigma, cv::BORDER_DEFAULT);
        }*/
    }
    // TODO: back-project filtered depth values to obtain vertex map
    void back_projection(){
        //can change the type just early ideas
        Vector4f vertex_map;

        
    }

    // TODO: compute normal vectors

    // TODO: apply vertex validity mask

    // TODO: transorm all to global

    // TODO: compute all multiscale

private:
    int num_levels;
    float bilateral_color_sigma;
    float bilateral_spatial_sigma;
    int depth_diameter;

    float fX;
    float fY;
    float cX;
    float cY;

    float *m_depthMap;
    BYTE *m_colorMap;
    Matrix4f m_trajectory;
    Matrix4f m_trajectoryInv;
    Matrix3f m_depthIntrinsics;
    unsigned int m_colorImageWidth;
    unsigned int m_colorImageHeight;
    unsigned int m_depthImageWidth;
    unsigned int m_depthImageHeight;
    CameraRefPoints *camera_reference_points;
    GlobalPoints *global_points;
    SurfaceLevelData *all_data;
};