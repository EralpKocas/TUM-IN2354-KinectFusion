//
// Created by ilteber on 12.07.21.
//

#ifndef KINECTFUSION_GPU_POSE_ESTIMATION_H
#define KINECTFUSION_GPU_POSE_ESTIMATION_H

#include "data_types.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "opencv2/core/cuda_types.hpp"


void point_to_plane( SurfaceLevelData* surf_data,cv::cuda::PtrStepSz<Vector3f> source, cv::cuda::PtrStepSz<Vector3f> dest, std::vector<Vector3f> normal, int level, Isometry3f& T);
__global__ void check_correspondence_validity(int level, int point, cv::cuda::PtrStepSz<Vector3f> vertex_map_predicted, cv::cuda::PtrStepSz<Vector3f> normal_map_predicted, bool& validity);
__global__ void get_global_vertex_map( cv::cuda::PtrStepSz<float> depth_map, cv::cuda::PtrStep<Vector3f> vertex_map);
__global__ void get_global_normal_map( cv::cuda::PtrStepSz<float> depth_map, cv::cuda::PtrStep<Vector3f> normal_map, cv::cuda::PtrStep<Vector3f> &global_normal_map,
                                       Matrix3f rotation, Vector3f translation, int width, int height, int level);
void pose_estimate(const std::vector<int>& iterations, ImageConstants*& imageConstants, ImageData* imageData, SurfaceLevelData* surf_data);

void init_global_map(cv::cuda::PtrStepSz<float> depth_map,
                     cv::cuda::PtrStep<Vector3f> vertex_map,
                     cv::cuda::PtrStep<Vector3f> normal_map,
                     cv::cuda::PtrStep<Vector3f>& global_vertex_map,
                     cv::cuda::PtrStep<Vector3f>& global_normal_map,
                     Matrix3f rotation, Vector3f translation,
                     int width, int height, int level);
void pose_estimate_helper(std::vector<int> iterations, int level,
                          cv::cuda::PtrStepSz<float> depth_map,
                          cv::cuda::PtrStepSz<Vector3f> vertex_map,
                          cv::cuda::PtrStepSz<Vector3f> vertex_map_predicted,
                          cv::cuda::PtrStepSz<Vector3f> normal_map_predicted,
                          cv::cuda::PtrStepSz<Vector3f> global_vertex_map,
                          SurfaceLevelData* surf_data,
                          Matrix3f &rotation, Vector3f &translation);

#endif //KINECTFUSION_GPU_POSE_ESTIMATION_H
