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
        float camera_x = ((float) pixel.x() - cX) / fX;
        float camera_y = ((float) pixel.y() - cY) / fY;
        // return or set to smth global pose * K^(-1) * u^.
        return rotation * Vector3f(camera_x, camera_y, 1.f) + translation;
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
                int width = image_properties->all_data[level].img_width;
                int height = image_properties->all_data[level].img_height;
                for(int i=0; i < width; i++)
                {
                    for(int j=0; j < height; j++)
                    {
                        Vector3f ray = calculate_pixel_raycast(Vector3f(float(i), float(j), 1.f), rotation, translation,
                                image_properties->all_data[level].curr_fX, image_properties->all_data[level].curr_fY,
                                image_properties->all_data[level].curr_cX, image_properties->all_data[level].curr_cY);

                    }
                }
            }
        }
private:

};

#endif
