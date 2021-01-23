#include <array>

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "Eigen.h"
#include "ceres/ceres.h"
#include "common.h"


class SurfaceMeasurement {

public:

    // initialize
    SurfaceMeasurement() { }

    bool init_pyramid(ImageProperties* image_properties)
    {
        // TODO: will initialize pyramid for given number of layers
        // TODO: need to change camera parameters with scaling level and need to add cv::pyrDown() so that values are resized accordingly.

        for(int i=0; i < num_levels; i++)
        {
            if(i==0) image_properties->all_data[i].curr_level_data = image_properties->m_depthMap;
            else cv::pyrDown(image_properties->all_data[i-1].curr_level_data,
                            image_properties->all_data[i].curr_level_data);

            auto scale = (float) ceres::pow(2, i);

            image_properties->all_data[i].img_width = (float) image_properties->m_colorImageWidth / scale;
            image_properties->all_data[i].img_height = (float) image_properties->m_colorImageHeight / scale;
            image_properties->all_data[i].curr_fX = image_properties->fX / scale;
            image_properties->all_data[i].curr_fY = image_properties->fY / scale;
            image_properties->all_data[i].curr_cX = image_properties->cX / scale;
            image_properties->all_data[i].curr_cY = image_properties->cY / scale;
        }
        return true;
    }

    Vector4f convert_homogeneous_vector(Vector3f vector_3d)
    {
        return Vector4f(vector_3d.x(), vector_3d.y(), vector_3d.z(), (float) 1.0);
    }

    bool init(ImageProperties* image_properties)
    {
        image_properties->all_data = new SurfaceLevelData[num_levels];
        num_levels = 3;
        bilateral_color_sigma = 1.;
        bilateral_spatial_sigma = 1.;
        depth_diameter = 3 * (int) bilateral_color_sigma;
        return init_pyramid(image_properties);
    }

    // compute bilateral filter
    void compute_bilateral_filter(ImageProperties* image_properties)
    {
        for(int i=0; i < num_levels; i++){
            cv::bilateralFilter(image_properties->all_data[i].curr_level_data, image_properties->all_data[i].curr_smoothed_data,
                    depth_diameter, bilateral_color_sigma, bilateral_spatial_sigma, cv::BORDER_DEFAULT);
        }
    }

    void helper_compute_vertex_map(ImageProperties* image_properties, int level, float fX, float fY, float cX, float cY)
    {
        int curr_width = (int) image_properties->all_data[level].img_width;
        int curr_height = (int) image_properties->all_data[level].img_height;
        int numWH = curr_width * curr_height;

        for(int i=0; i < numWH; i++){
            float currDepthValue =  image_properties->all_data[level].curr_smoothed_data.at<float>(i);
            if(currDepthValue == MINF){
                image_properties->all_data[level].vertex_map[i] = Vector3f(MINF, MINF, MINF);
                // TODO: figure out color!
                //imageProperties->all_data[level].curr_level_color[i] = Vector4uc(0, 0, 0, 0);
            }
            else{
                int pixel_y = i / curr_width;
                int pixel_x = i - pixel_y * curr_width;
                float camera_x = currDepthValue * ((float) pixel_x - cX) / fX;
                float camera_y = currDepthValue * ((float) pixel_y - cY) / fY;

                image_properties->all_data[level].vertex_map[i] = Vector3f(camera_x, camera_y, currDepthValue);
                // TODO: figure out color!
                //imageProperties->camera_reference_points[i].color = Vector4uc(imageProperties->m_colorMap[4*i],
                // imageProperties->m_colorMap[4*i+1], imageProperties->m_colorMap[4*i+2], imageProperties->m_colorMap[4*i+3]);
            }
        }
    }

    // TODO: back-project filtered depth values to obtain vertex map
    void compute_vertex_map(ImageProperties* image_properties){
        for(int i=0; i < num_levels; i++){
            helper_compute_vertex_map(image_properties, i, image_properties->all_data[i].curr_fX,
                    image_properties->all_data[i].curr_fY, image_properties->all_data[i].curr_cX,
                    image_properties->all_data[i].curr_cY);
        }
    }

    // TODO: compute normal vectors
    void helper_compute_normal_map(ImageProperties* image_properties, int level)
    {
        int curr_width = (int) image_properties->all_data[level].img_width;
        int curr_height = (int) image_properties->all_data[level].img_height;
        int numWH = curr_width * curr_height;

        for(int i=0; i < numWH; i++){
            Vector3f curr_vertex = image_properties->all_data[level].vertex_map[i];

            if(curr_vertex.z() == MINF){
                image_properties->all_data[level].vertex_map[i] = Vector3f(MINF, MINF, MINF);
                // TODO: figure out color if necessary (?)!
                //imageProperties->all_data[level].curr_level_color[i] = Vector4uc(0, 0, 0, 0);
            }
            else{
                int pixel_y = i / curr_width;
                int pixel_x = i - pixel_y * curr_width;

                int right_pixel = (pixel_x + 1) + pixel_y * curr_width;
                int bottom_pixel = pixel_x + (pixel_y + 1) * curr_width;


                Vector3f neigh_1 = Vector3f(image_properties->all_data[level].vertex_map[right_pixel].x() -
                                            image_properties->all_data[level].vertex_map[i].x(),
                                            image_properties->all_data[level].vertex_map[right_pixel].y() -
                                            image_properties->all_data[level].vertex_map[i].y(),
                                            image_properties->all_data[level].vertex_map[right_pixel].z() -
                                            image_properties->all_data[level].vertex_map[i].z());

                Vector3f neigh_2 = Vector3f(image_properties->all_data[level].vertex_map[bottom_pixel].x() -
                                            image_properties->all_data[level].vertex_map[i].x(),
                                            image_properties->all_data[level].vertex_map[bottom_pixel].y() -
                                            image_properties->all_data[level].vertex_map[i].y(),
                                            image_properties->all_data[level].vertex_map[bottom_pixel].z() -
                                            image_properties->all_data[level].vertex_map[i].z());

                Vector3f cross_prod = neigh_1.cross(neigh_2);
                cross_prod.normalize();

                image_properties->all_data[level].normal_map[i] = cross_prod;
                // TODO: figure out color if necessary (?)!
                //imageProperties->camera_reference_points[i].color = Vector4uc(imageProperties->m_colorMap[4*i],
                // imageProperties->m_colorMap[4*i+1], imageProperties->m_colorMap[4*i+2], imageProperties->m_colorMap[4*i+3]);
            }
        }
    }

    void compute_normal_map(ImageProperties* image_properties){
        for(int i=0; i < num_levels; i++){
            helper_compute_normal_map(image_properties, i);
        }
    }

    // TODO: apply vertex validity mask
    // TODO: check if already applied in helper_compute_vertex_map.

    // TODO: transorm all to global

    // TODO: compute all multiscale
    void surface_measurement_pipeline(ImageProperties* image_properties)
    {
        if(!init(image_properties))
        {
            std::cout << "Failed to initialize the SurfaceMeasurement step!\nCheck multiscale pyramid creation!" << std::endl;
        }
        compute_bilateral_filter(image_properties);
        compute_vertex_map(image_properties);
        compute_normal_map(image_properties);
    }

private:
    int num_levels;
    float bilateral_color_sigma;
    float bilateral_spatial_sigma;
    int depth_diameter;

    /*float fX;
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
    SurfaceLevelData *all_data;*/
};