//
// Created by ilteber on 05.07.21.
//

#ifndef TUM_IN2354_KINECTFUSION_SURFACE_RECONSTRUCTION_H
#define TUM_IN2354_KINECTFUSION_SURFACE_RECONSTRUCTION_H

#include "data_types.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "opencv2/core/cuda_types.hpp"

////HELPER FUNCTIONS
__device__ float calculateLambda(Matrix4f depth_ext,
                                 Matrix4f m_traj_inv,
                                 Matrix3f depth_intrinsics,
                                 Matrix3f intrinsics, Vector3f p);
//
__device__ Vector2i perspective_projection(Matrix4f depth_ext,
                                           Matrix4f m_traj_inv,
                                           Matrix3f depth_intrinsics, Vector3f p);
//
__device__ float calculateSDF_truncation(float truncation_distance, float sdf);
//λ = ||K^-1*x||2
__device__ float calculateCurrentTSDF(Matrix4f pose_traj, Matrix4f depth_ext,
                                      Matrix4f m_traj_inv, Matrix3f depth_intrinsics, float depth,
                                      Matrix3f intrinsics, Vector3f p, float truncation_distance);
//
//
// calculate weighted running tsdf average
__device__ float calculateWeightedTSDF(int current_weight, float current_tsdf, int new_weight, float new_tsdf);

// TODO: calculate weighted running weight average
__device__ int calculateWeightedAvgWeight(int current_weight, int new_weight);

// TODO: truncate updated weight
__device__ int calculateTruncatedWeight(int weighted_avg, int some_value);

__device__ Vector4uc calculateWeightedColorUpdate(int current_weight, Vector4uc curr_color,
                                                  int new_weight, Vector4uc new_color);
__global__ void updateSurfaceReconstructionGlobal(ImageConstants*& imageConstants,
                                                  cv::cuda::PtrStepSz<float> tsdf_values,
                                                  cv::cuda::PtrStepSz<int> tsdf_weight,
                                                  cv::cuda::PtrStepSz<Vector4uc> tsdf_color,
                                                  cv::cuda::PtrStepSz<Vector4uc> color_map,
                                                  cv::cuda::PtrStepSz<float> depth_map,
                                                  int width, int height,
                                                  float voxel_scale,
                                                  int volume_size,
                                                  Matrix4f depth_ext,
                                                  Matrix4f m_traj,
                                                  Matrix4f m_traj_inv,
                                                  Matrix3f depth_intrinsics,
                                                  Matrix3f depth_intrinsics_inv,
                                                  Matrix4f pose_traj,
                                                  float truncation_distance);

void updateSurfaceReconstruction(Pose* pose,ImageConstants* imageConstants,
                                       ImageData* imageData, SurfaceLevelData* surf_data,
                                        GlobalVolume* global_volume);


//private:
//
//float depth_margin;                 //μ
//float *m_depthMap;                  //Rk
//ImageProperties* imageProperties;

#endif //TUM_IN2354_KINECTFUSION_SURFACE_RECONSTRUCTION_H
