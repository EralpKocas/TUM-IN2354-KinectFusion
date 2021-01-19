#include <array>

#include <opencv2/opencv.hpp>
#include "Eigen.h"
#include "ceres/ceres.h"


class SurfaceMeasurement {

public:

    // initialize
    //SurfaceMeasurement() : { }

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

        //cv::pyrDown(all_data*, , , cv::BORDER_DEFAULT);
        return true;
    }

    Vector4f convert_homogeneous_vector(Vector3f vector_3d)
    {
        return Vector4f(vector_3d.x(), vector_3d.y(), vector_3d.z(), (float) 1.0);
    }

    // compute 3d camera reference points
    void compute_camera_ref_points()
    {
        int numWH = m_depthImageWidth * m_depthImageHeight;

        for(int i=0; i < numWH; i++){
            if(m_depthMap[i] == MINF){
                camera_reference_points[i].position = Vector3f(MINF, MINF, MINF);
                global_points[i].color = Vector4uc(0,0,0,0);
            }
            else{
                int pixel_y = i / m_depthImageWidth;
                int pixel_x = i - pixel_y * m_depthImageWidth;
                float currDepthValue = m_depthMap[i];
                float camera_x = currDepthValue * ((float) pixel_x - cX) / fX;
                float camera_y = currDepthValue * ((float) pixel_y - cY) / fY;

                camera_reference_points[i].position = Vector3f(camera_x, camera_y, currDepthValue);
                camera_reference_points[i].color = Vector4uc(m_colorMap[4*i], m_colorMap[4*i+1], m_colorMap[4*i+2], m_colorMap[4*i+3]);
            }
        }
        for(int i=0; i < numWH; i++) {
            if (m_depthMap[i] != MINF) {
                std::cout << "first camera ref point at i: \n" << i << std::endl;
                std::cout << "curr depth: \n" << m_depthMap[i] << std::endl;
                std::cout << "camera_reference_points: \n" << camera_reference_points[i].position << std::endl;
                break;
            }
        }
    }

    // compute global 3D points
    void compute_global_points()
    {
        int numWH = m_depthImageWidth * m_depthImageHeight;

        for(int i=0; i < numWH; i++) {
            if (m_depthMap[i] == MINF) {
                global_points[i].position = Vector4f(MINF, MINF, MINF, MINF);
                global_points[i].color = Vector4uc(0, 0, 0, 0);
            } else {
                Vector4f camera_ref_vector = Vector4f(camera_reference_points[i].position.x(),
                                                      camera_reference_points[i].position.y(),
                                                      camera_reference_points[i].position.z(),
                                                      (float) 1.0);

                Vector4f global_point = m_trajectory * camera_ref_vector;

                global_points[i].position = global_point;
                global_points[i].color = camera_reference_points[i].color;
            }
        }
        for(int i=0; i < numWH; i++) {
            if (m_depthMap[i] != MINF) {
                std::cout << "first global point at i: \n" << i << std::endl;
                std::cout << "m_trajectory: \n" << m_trajectory << std::endl;
                std::cout << "global: \n" << global_points[i].position << std::endl;
                break;
            }
        }
    }

    // I am not sure whether we'll need this for this step.
    Vector2f perspective_projection(Vector3f p)
    {
        return Vector2f(p.x() / p.z(), p.y() / p.z());
    }


    // compute bilateral filter
    // TODO: does not work unless we change all_data.curr_level_data to a proper array.
    void compute_bilateral_filter()
    {
        for(int i=0; i<num_levels;i++){
            cv::bilateralFilter(all_data[i].curr_level_data, all_data[i].curr_smoothed_data,
                    depth_diameter, bilateral_color_sigma, bilateral_spatial_sigma, cv::BORDER_DEFAULT);
        }
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