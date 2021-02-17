#pragma once

#ifndef KINECTFUSION_SURFACE_MEASUREMENT_H
#define KINECTFUSION_SURFACE_MEASUREMENT_H

#include <array>

//#include <opencv4/opencv2/opencv.hpp>
//#include <opencv2/opencv.hpp>
/*#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Eigen.h"
#include "ceres/ceres.h"*/
#include <common.h>


class SurfaceMeasurement {

public:

    // initialize
    SurfaceMeasurement() { }

    bool init_pyramid(ImageProperties*& image_properties)
    {

        for(int i=0; i < image_properties->num_levels; i++)
        {

            auto scale = (float) ceres::pow(2, i);

            image_properties->all_data[i].img_width = (float) image_properties->m_colorImageWidth / scale;
            image_properties->all_data[i].img_height = (float) image_properties->m_colorImageHeight / scale;
            image_properties->all_data[i].curr_fX = image_properties->fX / scale;
            image_properties->all_data[i].curr_fY = image_properties->fY / scale;
            image_properties->all_data[i].curr_cX = image_properties->cX / scale;
            image_properties->all_data[i].curr_cY = image_properties->cY / scale;

            /*std::cout << "i: \n" << i << std::endl;
            std::cout << "scale: \n" << scale << std::endl;
            std::cout << "image_properties->all_data[i].img_width: \n" << image_properties->all_data[i].img_width << std::endl;
            std::cout << "image_properties->all_data[i].img_height: \n" << image_properties->all_data[i].img_height << std::endl;
            std::cout << "image_properties->all_data[i].curr_fX: \n" << image_properties->all_data[i].curr_fX << std::endl;
            std::cout << "image_properties->all_data[i].curr_fY: \n" << image_properties->all_data[i].curr_fY << std::endl;
            std::cout << "image_properties->all_data[i].curr_cX: \n" << image_properties->all_data[i].curr_cX << std::endl;
            std::cout << "image_properties->all_data[i].curr_cY: \n" << image_properties->all_data[i].curr_cY << std::endl;
            */

            if(i==0){
                image_properties->all_data[i].curr_level_data = image_properties->m_depthMap;
                compute_bilateral_filter(image_properties);
            }
            else{
                image_properties->all_data[i].curr_smoothed_data = cv::Mat(image_properties->all_data[i].img_width,
                        image_properties->all_data[i].img_height, CV_32F);
                cv::pyrDown(image_properties->all_data[i-1].curr_smoothed_data,
                        image_properties->all_data[i].curr_smoothed_data);
            }

            //std::cout << "image_properties->all_data[i].curr_level_data: \n" << image_properties->all_data[i].curr_level_data << std::endl;

            image_properties->all_data[i].vertex_map = std::vector<Vector3f>(image_properties->all_data[i].img_width *
                                                                                    image_properties->all_data[i].img_height);
            image_properties->all_data[i].normal_map = std::vector<Vector3f>(image_properties->all_data[i].img_width *
                                                                             image_properties->all_data[i].img_height);
            image_properties->all_data[i].vertex_map_predicted = std::vector<Vector3f>(image_properties->all_data[i].img_width *
                                                                             image_properties->all_data[i].img_height);
            image_properties->all_data[i].normal_map_predicted = std::vector<Vector3f>(image_properties->all_data[i].img_width *
                                                                             image_properties->all_data[i].img_height);

        }
        return true;
    }

    Vector4f convert_homogeneous_vector(Vector3f vector_3d)
    {
        return Vector4f(vector_3d.x(), vector_3d.y(), vector_3d.z(), (float) 1.0);
    }

    bool init(ImageProperties*& image_properties)
    {
        bilateral_color_sigma = 1.;
        bilateral_spatial_sigma = 1.;
        depth_diameter = 3 * (int) bilateral_color_sigma;
        image_properties->all_data = new SurfaceLevelData[image_properties->num_levels];
        return init_pyramid(image_properties);
    }

    // compute bilateral filter
    void compute_bilateral_filter(ImageProperties*& image_properties)
    {
        /*for(int i=0; i < image_properties->num_levels; i++){
            cv::bilateralFilter(image_properties->all_data[i].curr_level_data, image_properties->all_data[i].curr_smoothed_data,
                    depth_diameter, bilateral_color_sigma, bilateral_spatial_sigma, cv::BORDER_DEFAULT);
        }*/
        cv::bilateralFilter(image_properties->all_data[0].curr_level_data, image_properties->all_data[0].curr_smoothed_data,
                            depth_diameter, bilateral_color_sigma, bilateral_spatial_sigma, cv::BORDER_DEFAULT);
    }

    void helper_compute_vertex_map(ImageProperties*& image_properties, int level, float fX, float fY, float cX, float cY)
    {
        int curr_width = (int) image_properties->all_data[level].img_width;
        int curr_height = (int) image_properties->all_data[level].img_height;
        //int numWH = curr_width * curr_height;

        for(int i=0; i < curr_height; i++){
            for(int j=0; j < curr_width; j++){
                float currDepthValue =  image_properties->all_data[level].curr_smoothed_data.at<float>(i, j);
                if(currDepthValue == MINF || isnan(currDepthValue)){
                    image_properties->all_data[level].vertex_map[j + i * curr_width] = Vector3f(MINF, MINF, MINF);
                    //imageProperties->all_data[level].curr_level_color[i] = Vector4uc(0, 0, 0, 0);
                }
                else{
                    //int pixel_y = i / curr_width;
                    //int pixel_x = i - pixel_y * curr_width;
                    int pixel_x = j;
                    int pixel_y = i;
                    float camera_x = currDepthValue * ((float) pixel_x - cX) / fX;
                    float camera_y = currDepthValue * ((float) pixel_y - cY) / fY;

                    image_properties->all_data[level].vertex_map[j + i * curr_width] = Vector3f(camera_x, camera_y, currDepthValue);

                    //imageProperties->camera_reference_points[i].color = Vector4uc(imageProperties->m_colorMap[4*i],
                    // imageProperties->m_colorMap[4*i+1], imageProperties->m_colorMap[4*i+2], imageProperties->m_colorMap[4*i+3]);
                }
            }
        }
    }

    // back-project filtered depth values to obtain vertex map
    void compute_vertex_map(ImageProperties*& image_properties){
        for(int i=0; i < image_properties->num_levels; i++){
            helper_compute_vertex_map(image_properties, i, image_properties->all_data[i].curr_fX,
                    image_properties->all_data[i].curr_fY, image_properties->all_data[i].curr_cX,
                    image_properties->all_data[i].curr_cY);
        }
    }

    // compute normal vectors
    void helper_compute_normal_map(ImageProperties*& image_properties, int level)
    {
        int curr_width = (int) image_properties->all_data[level].img_width;
        int curr_height = (int) image_properties->all_data[level].img_height;
        //int numWH = curr_width * curr_height;

        for(int i=0; i < curr_height; i++){
            for(int j=0; j < curr_width; j++) {
                Vector3f curr_vertex = image_properties->all_data[level].vertex_map[j + i * curr_width];

                if (curr_vertex.z() == MINF) {
                    image_properties->all_data[level].normal_map[j + i * curr_width] = Vector3f(MINF, MINF, MINF);
                    //imageProperties->all_data[level].curr_level_color[i] = Vector4uc(0, 0, 0, 0);
                } else {
                    int pixel_y = i;
                    int pixel_x = j;

                    int right_pixel = (pixel_x + 1) + pixel_y * curr_width;
                    int left_pixel = (pixel_x - 1) + pixel_y * curr_width;
                    int bottom_pixel = pixel_x + (pixel_y + 1) * curr_width;
                    int upper_pixel = pixel_x + (pixel_y - 1) * curr_width;

                    Vector3f neigh_1 = Vector3f(image_properties->all_data[level].vertex_map[left_pixel].x() -
                                                image_properties->all_data[level].vertex_map[right_pixel].x(),
                                                image_properties->all_data[level].vertex_map[left_pixel].y() -
                                                image_properties->all_data[level].vertex_map[right_pixel].y(),
                                                image_properties->all_data[level].vertex_map[left_pixel].z() -
                                                image_properties->all_data[level].vertex_map[right_pixel].z());

                    Vector3f neigh_2 = Vector3f(image_properties->all_data[level].vertex_map[upper_pixel].x() -
                                                image_properties->all_data[level].vertex_map[bottom_pixel].x(),
                                                image_properties->all_data[level].vertex_map[upper_pixel].y() -
                                                image_properties->all_data[level].vertex_map[bottom_pixel].y(),
                                                image_properties->all_data[level].vertex_map[upper_pixel].z() -
                                                image_properties->all_data[level].vertex_map[bottom_pixel].z());

                    Vector3f cross_prod = neigh_1.cross(neigh_2);
                    cross_prod.normalize();
                    if (cross_prod.z() > 0) cross_prod *= -1;
                    image_properties->all_data[level].normal_map[j + i * curr_width] = cross_prod;

                    /*image_properties->camera_reference_points[i].color = Vector4uc(image_properties->m_colorMap[4*i],
                                                                                   image_properties->m_colorMap[4*i+1],
                                                                                   image_properties->m_colorMap[4*i+2],
                                                                                   image_properties->m_colorMap[4*i+3]);*/
                }
            }
        }
    }

    void compute_normal_map(ImageProperties*& image_properties){
        for(int i=0; i < image_properties->num_levels; i++){
            helper_compute_normal_map(image_properties, i);
        }
    }

    void surface_measurement_pipeline(ImageProperties*& image_properties)
    {
        if(!init(image_properties))
        {
            std::cout << "Failed to initialize the SurfaceMeasurement step!\nCheck multiscale pyramid creation!" << std::endl;
        }
        compute_vertex_map(image_properties);
        compute_normal_map(image_properties);

    }

private:

    float bilateral_color_sigma;
    float bilateral_spatial_sigma;
    int depth_diameter;

};

#endif
