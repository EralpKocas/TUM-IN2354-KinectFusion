//
// Created by eralpkocas on 26.06.21.
//

#ifndef TUM_IN2354_KINECTFUSION_SURFACE_MEASUREMENT_H
#define TUM_IN2354_KINECTFUSION_SURFACE_MEASUREMENT_H

#include "data_types.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "opencv2/core/cuda_types.hpp"

__global__ void helper_compute_vertex_map(SurfaceLevelData* surf_data, cv::cuda::PtrStepSz<float> depth_map,
                                          ImageConstants img_constants,
                                          cv::cuda::PtrStep<Vector3f> vertex_map, float fX, float fY,
                                          float cX, float cY, int width, int height, int level);

__global__ void helper_compute_normal_map(SurfaceLevelData* surf_data, cv::cuda::PtrStepSz<Vector3f> vertex_map,
                                          cv::cuda::PtrStep<Vector3f> normal_map,
                                          int width, int height, int level);

bool init_multiscale(SurfaceLevelData* surf_data, ImageData img_data, ImageConstants img_constants);

void compute_vertex_map(SurfaceLevelData* surf_data, ImageConstants img_constants);

void compute_normal_map(SurfaceLevelData* surf_data);

void surface_measurement_pipeline(SurfaceLevelData* surf_data, ImageData img_data, ImageConstants img_constants);

#endif //TUM_IN2354_KINECTFUSION_SURFACE_MEASUREMENT_H
