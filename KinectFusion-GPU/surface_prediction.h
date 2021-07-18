//
// Created by ilteber on 14.07.21.
//

#ifndef KINECTFUSION_GPU_SURFACE_PREDICTION_H
#define KINECTFUSION_GPU_SURFACE_PREDICTION_H
#include "data_types.h"
#include "surface_prediction.h"

//WARNING: Volume stuff might not have converted to CUDA right.

__device__ Vector3f calculate_pixel_ray_cast(Matrix3f rotation, Vector3f translation, ImageConstants imageConstants);


__device__ Vector3f calculate_ray_cast_dir(Vector3f eye, Vector3f current_ray);

// P = O + t * R where R is normalized ray direction and O is eye, translation vector in our case.
// Then, t = (P - O) / R
__device__ float calculate_search_length(Vector3f eye, Vector3f ray_dir);

__device__ bool gridInVolume(Volume* global_volume, Vector3f curr_grid);

__device__ float calculate_trilinear_interpolation(GlobalVolume* global_volume, Vector3f p);

__global__ void helper_compute_normal_map(int width, int height);

void predict_surface(ImageConstants image_constants, SurfaceLevelData* surf_data, GlobalVolume* global_volume,
                     Matrix3f rotation, Vector3f translation, int width, int height, int level);

#endif //KINECTFUSION_GPU_SURFACE_PREDICTION_H
