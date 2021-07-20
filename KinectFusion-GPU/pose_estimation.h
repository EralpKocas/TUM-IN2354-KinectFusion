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

__device__ void write_linear_eq(MatrixXf& A, VectorXf& b, int i, bool fill_zero,
                                const Vector3f& curr_global_vertex,
                                const Vector3f& prev_global_vertex,
                                const Vector3f& prev_global_normal);

__global__ void form_linear_eq_new(int width, int height,
                                   cv::cuda::PtrStepSz<Vector3f> curr_frame_vertex,
                                   cv::cuda::PtrStepSz<Vector3f> curr_frame_normal,
                                   cv::cuda::PtrStepSz<Vector3f> prev_global_vertex,
                                   cv::cuda::PtrStepSz<Vector3f> prev_global_normal,
                                   MatrixXf& A, VectorXf& b,
                                   Matrix3f &curr_rotation, Vector3f &curr_translation,
                                   Matrix3f &prev_rotation_inv, Vector3f &prev_translation,
                                   float fX, float fY);

void point_to_plane_new( cv::cuda::GpuMat& curr_frame_vertex,
                         cv::cuda::GpuMat& curr_frame_normal,
                         cv::cuda::GpuMat& prev_global_vertex,
                         cv::cuda::GpuMat& prev_global_normal,
                         int width, int height,
                         MatrixXf A, VectorXf b,
                         Matrix3f curr_rotation, Vector3f curr_translation,
                         Matrix3f prev_rotation_inv, Vector3f prev_translation,
                         float fX, float fY,
                         Isometry3f& T);

void pose_estimate_helper_new(cv::cuda::GpuMat& curr_frame_vertex,
                              cv::cuda::GpuMat& curr_frame_normal,
                              cv::cuda::GpuMat& prev_global_vertex,
                              cv::cuda::GpuMat& prev_global_normal,
                              int width, int height,
                              MatrixXf A, VectorXf b,
                              Matrix3f& curr_rotation, Vector3f& curr_translation,
                              Matrix3f prev_rotation_inv, Vector3f prev_translation,
                              float fX, float fY);

void pose_estimate_new(const std::vector<int>&  iterations,
                       SurfaceLevelData* surf_data,
                       Pose* pose_struct);



__global__ void get_global_vertex_map( cv::cuda::PtrStepSz<float> depth_map,
                                       cv::cuda::PtrStep<Vector3f> vertex_map,
                                       cv::cuda::PtrStep<Vector3f> normal_map,
                                       Matrix3f rotation, Vector3f translation,
                                       int width, int height,
                                       cv::cuda::PtrStep<Vector3f> global_vertex_map,
                                       cv::cuda::PtrStep<Vector3f> camera_vertex_map);
__global__ void get_global_normal_map( cv::cuda::PtrStepSz<float> depth_map,
                                       cv::cuda::PtrStep<Vector3f> normal_map,
                                       Matrix3f rotation, Vector3f translation,
                                       int width, int height,
                                       cv::cuda::PtrStep<Vector3f> global_normal_map);
__global__ void check_correspondence_validity(int width, int height,
                                              cv::cuda::PtrStepSz<Vector3f> vertex_map_predicted,
                                              cv::cuda::PtrStepSz<Vector3f> normal_map_predicted,
                                              cv::cuda::PtrStepSz<Vector3f> global_vertex_map,
                                              cv::cuda::PtrStepSz<Vector3f> global_normal_map,
                                              cv::cuda::PtrStepSz<float> depth_map,
                                              bool& validity);
__global__ void form_linear_eq(int width, int height,
                               cv::cuda::PtrStepSz<Vector3f> source,
                               cv::cuda::PtrStepSz<Vector3f> normal,
                               cv::cuda::PtrStepSz<Vector3f> dest,
                               Eigen::Matrix<float, 6, 6, Eigen::RowMajor> &A,
                               Eigen::Matrix<float, 6, 1> &b);

void point_to_plane( cv::cuda::GpuMat source,
                     cv::cuda::GpuMat dest,
                     cv::cuda::GpuMat normal,
                     cv::cuda::GpuMat global_normal_map,
                     cv::cuda::GpuMat depth_map,
                     int width, int height,
                     Isometry3f& T);

void pose_estimate(const std::vector<int>& iterations, ImageConstants* imageConstants,
                   ImageData* imageData, SurfaceLevelData* surf_data, Pose* pose_struct);

void init_global_map(cv::cuda::GpuMat depth_map,
                     cv::cuda::GpuMat vertex_map,
                     cv::cuda::GpuMat normal_map,
                     cv::cuda::GpuMat global_vertex_map,
                     cv::cuda::GpuMat camera_vertex_map,
                     cv::cuda::GpuMat global_normal_map,
                     Matrix3f rotation, Vector3f translation,
                     int width, int height, int level);

void pose_estimate_helper( int iteration,
                           cv::cuda::GpuMat depth_map,
                           cv::cuda::GpuMat vertex_map,
                           cv::cuda::GpuMat vertex_map_predicted,
                           cv::cuda::GpuMat normal_map_predicted,
                           cv::cuda::GpuMat global_vertex_map,
                           cv::cuda::GpuMat global_normal_map,
                           int width, int height,
                           Matrix3f &rotation, Vector3f &translation);

#endif //KINECTFUSION_GPU_POSE_ESTIMATION_H
