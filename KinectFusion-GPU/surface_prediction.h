//
// Created by eralpkocas on 18.07.21.
//
#include "data_types.h"
#ifndef KINECTFUSION_GPU_SURFACE_PREDICTION_H
#define KINECTFUSION_GPU_SURFACE_PREDICTION_H

void surface_prediction(SurfaceLevelData* surf_data, GlobalVolume global_volume, Pose pose);

__global__ void predict_surface(GlobalVolume global_volume, Pose pose,
                                cv::cuda::PtrStep<Vector3f> vertex_map,
                                cv::cuda::PtrStep<Vector3f> normal_map,
                                cv::cuda::PtrStep<Vector4uc> color_map,
                                float fX, float fY, float cX, float cY,
                                int width, int height, int level);

__global__ void helper_compute_normal_map(int width, int height);

__device__ float calculate_trilinear_interpolation(GlobalVolume* global_volume, Vector3f p);

__device__ bool gridInVolume(GlobalVolume* global_volume, Vector3f curr_grid);

__device__ float calculate_search_length(Vector3f eye, Vector3f ray_dir);

__device__ Vector3f calculate_ray_cast_dir(Vector3f eye, Vector3f current_ray);

__device__ Vector3f calculate_pixel_ray_cast(Matrix3f rotation, Vector3f translation, ImageConstants imageConstants);
#endif //KINECTFUSION_GPU_SURFACE_PREDICTION_H
