//
// Created by ilteber on 12.07.21.
//

#ifndef KINECTFUSION_GPU_POSE_ESTIMATION_H
#define KINECTFUSION_GPU_POSE_ESTIMATION_H

#include "data_types.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "opencv2/core/cuda_types.hpp"

__device__ Vector2i backproject_vertex(const Vector3f& curr_global_vertex,
                                              const Matrix3f& prev_rotation_inv,
                                              const Vector3f& prev_translation,
                                              float fX, float fY);

__device__ bool isVertDistValid(const Vector3f& curr_global_vertex,
                                const Vector3f& prev_global_vertex,
                                float distance_threshold);

__device__ bool isNormAngleValid(const Vector3f& curr_global_normal,
                                 const Vector3f& prev_global_normal,
                                 float angle_threshold);

__device__ void write_linear_eq(float* A, float* b, int i, bool fill_zero,
                                const Vector3f& curr_global_vertex,
                                const Vector3f& prev_global_vertex,
                                const Vector3f& prev_global_normal);

__global__ void form_linear_eq_new(int width, int height,
                                   cv::cuda::PtrStepSz<Vector3f> curr_frame_vertex,
                                   cv::cuda::PtrStepSz<Vector3f> curr_frame_normal,
                                   cv::cuda::PtrStepSz<Vector3f> prev_global_vertex,
                                   cv::cuda::PtrStepSz<Vector3f> prev_global_normal,
                                   Matrix3f curr_rotation, Vector3f curr_translation,
                                   Matrix3f prev_rotation_inv, Vector3f prev_translation,
                                   float fX, float fY, float* d_A, float* d_b);

void point_to_plane_new( cv::cuda::GpuMat& curr_frame_vertex,
                         cv::cuda::GpuMat& curr_frame_normal,
                         cv::cuda::GpuMat& prev_global_vertex,
                         cv::cuda::GpuMat& prev_global_normal,
                         int width, int height,
                         Matrix3f curr_rotation, Vector3f curr_translation,
                         Matrix3f prev_rotation_inv, Vector3f prev_translation,
                         float fX, float fY,
                         Isometry3f& T);

void pose_estimate_helper_new(cv::cuda::GpuMat& curr_frame_vertex,
                              cv::cuda::GpuMat& curr_frame_normal,
                              cv::cuda::GpuMat& prev_global_vertex,
                              cv::cuda::GpuMat& prev_global_normal,
                              int width, int height,
                              Matrix3f& curr_rotation, Vector3f& curr_translation,
                              Matrix3f prev_rotation_inv, Vector3f prev_translation,
                              float fX, float fY);

void pose_estimate_new(const std::vector<int>&  iterations,
                       SurfaceLevelData* surf_data,
                       Pose* pose_struct);

#endif //KINECTFUSION_GPU_POSE_ESTIMATION_H
