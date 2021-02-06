#pragma once

#ifndef KINECTFUSION_SURFACE_PREDICTION_H
#define KINECTFUSION_SURFACE_PREDICTION_H

#include <array>

//#include <opencv4/opencv2/opencv.hpp>
//#include <opencv2/opencv.hpp>
/*#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Eigen.h"
#include "ceres/ceres.h"*/
#include <common.h>


class SurfacePrediction {

public:
    SurfacePrediction() {}

    void init()
    {

    }

    // TODO: define a function to calculate raycast of a pixel
    Vector3f calculate_pixel_raycast(Vector3f pixel, Matrix3f rotation, Vector3f translation,
            float fX, float fY, float cX, float cY)
    {
        float camera_x = ((float) pixel.x() - cX) / fX;  // image to camera
        float camera_y = ((float) pixel.y() - cY) / fY;  // image to camera
        return rotation * Vector3f(camera_x, camera_y, 1.f) + translation;  // camera to global
    }

    Vector3f calculate_raycast_dir(Vector3f eye, Vector3f current_ray)
    {
        return (eye - current_ray).normalized();
    }

    // P = O + t * R where R is normalized ray direction and O is eye, translation vector in our case.
    // Then, t = (P - O) / R
    float calculate_search_length(Vector3f eye, Vector3f pixel, Vector3f ray_dir)
    {
        float t_x = (pixel.x() - eye.x()) / ray_dir.x();
        float t_y = (pixel.y() - eye.y()) / ray_dir.y();
        float t_z = (pixel.z() - eye.z()) / ray_dir.z();

        return fmax(fmax(fabs(t_x), fabs(t_y)), fabs(t_z));
    }

    bool gridInVolume(ImageProperties*& image_properties, Vector3f curr_grid)
    {
        int dx = image_properties->global_tsdf->getDimX();
        int dy = image_properties->global_tsdf->getDimY();
        int dz = image_properties->global_tsdf->getDimZ();

        return !(curr_grid.x() < 1 || curr_grid.x() >= dx - 1 ||
                 curr_grid.y() < 1 || curr_grid.y() >= dy - 1 ||
                 curr_grid.z() < 1 || curr_grid.z() >= dz - 1);
    }
    // TODO: apply marching steps for per pixel u from minimum depth until finding a surface
        // stop conditions:
            // 1. when zero crossing is found
            // 2. -ve to +ve -> back-face is found
            // 3. if exceeds the working volume
            // 2 and 3 result as non-surface measurement at pixel u
            // TODO: implement conditions in a loop to understand continue to next pixel or not.
            // TODO: if not, calculate necessary vertex and normal maps.
        // for points very close to surface interface where F_{k}(p) = 0
            // surface normal for pixel u along which p -> compute directly using numerical derivative of F_{k}, SDF.
                // TODO: equation is on paper (Eq. 14)

        // TODO: understand and decide how to implement min and max sensor range (in paper -> [0.4, 8] meters)
        // TODO: understand the ray skipping
            // so far: decide on a step size to skip ray, maximum < μ.
                // around F(p) = 0 -> good approximation to true signed distance.

        // TODO: obtain higher quality intersections around the found intersections of SDF.
        // TODO: predicted vertex and normal maps are computed at interpolated location in the global frame.

        void predict_surface(ImageProperties*& image_properties)
        {
            for(int level=0; level < image_properties->num_levels; level++)
            {
                Vector3f translation = image_properties->m_depthExtrinsics.block(0, 3, 3, 1);
                Matrix3f rotation = image_properties->m_depthExtrinsics.block(0, 0, 3, 3);
                float truncation_distance = image_properties->truncation_distance;
                int width = (int) image_properties->all_data[level].img_width;
                int height = (int) image_properties->all_data[level].img_height;
                for(int i=0; i < width; i++)
                {
                    for(int j=0; j < height; j++)
                    {
                        // +0.5 for reaching pixel centers
                        Vector3f pixel_ray = calculate_pixel_raycast(Vector3f(float(i+0.5), float(j+0.5), 1.f), rotation, translation,
                                image_properties->all_data[level].curr_fX, image_properties->all_data[level].curr_fY,
                                image_properties->all_data[level].curr_cX, image_properties->all_data[level].curr_cY);
                        Vector3f ray_dir = calculate_raycast_dir(translation, pixel_ray);

                        float t = calculate_search_length(translation, pixel_ray, ray_dir);  // t

                        Vector3f pixel_grid = image_properties->global_tsdf->compute_grid(pixel_ray);
                        Vector3f ray_dir_grid = image_properties->global_tsdf->compute_grid(ray_dir);
                        Vector3f eye_grid = image_properties->global_tsdf->compute_grid(translation);

                        float tsdf = image_properties->global_tsdf->
                                get((int) eye_grid.x(), (int) eye_grid.y(), (int) eye_grid.z()).tsdf_distance_value;

                        float prev_tsdf = tsdf;
                        for(int step=0; (float) step < t; step+=truncation_distance)
                        {
                            Vector3f curr_grid = eye_grid + (float) step * ray_dir_grid;

                            if(!gridInVolume(image_properties, curr_grid)) continue;

                            float curr_tsdf = image_properties->global_tsdf->
                                    get((int) curr_grid.x(), (int) curr_grid.y(), (int) curr_grid.z()).tsdf_distance_value;

                            if(prev_tsdf < 0.f && tsdf > 0.f) break;  // zero-crossing from behind

                            if(prev_tsdf > 0.f && tsdf < 0.f)
                            {
                                float t_star = 0.f;  // t_star = t - ((step_size * prev_tsdf) / (curr_tsdf - prev_tsdf))
                            }

                            prev_tsdf = curr_tsdf;
                        }

                    }
                }
            }
        }

private:

};

#endif
