//
// Created by ilteber on 05.07.21.
//
#include "data_types.h"

#include "common_functions.h"
#include "surface_reconstruction.h"

////HELPER FUNCTIONS
__device__ float calculateLambda( Matrix4f depth_ext,
                                  Matrix4f m_traj,
                                  Matrix3f depth_intrinsics, Matrix3f intrinsicsInv, Vector3f p){
    Vector2i projected = perspective_projection(depth_ext, m_traj, depth_intrinsics, p);
    Vector3f dot_p = Vector3f(projected.x(), projected.y(), 1.0f);
    return (intrinsicsInv * dot_p).norm();
}
//

__device__ Vector2i perspective_projection(Matrix4f depth_ext,
                                           Matrix4f m_traj,
                                           Matrix3f depth_intrinsics, Vector3f p)
{
    Vector4f p_temp = Vector4f(p.x(), p.y(), p.z(), 1.0);
    Matrix4f identity = Matrix4f::Zero();
    identity.block<3, 3>(0, 0) = Matrix3f::Identity();
    Vector3f p2 = depth_intrinsics * identity.block<3, 4>(0, 0) * depth_ext * m_traj * p_temp;
    return Vector2i((int) round(p2.x() / p2.z()), (int) round(p2.y() / p2.z()));
}
//
//
__device__ float calculateSDF_truncation(float truncation_distance, float sdf){
    if (sdf >= -truncation_distance) {
        return fmin(1.f, sdf / truncation_distance) * (sdf < 0.f ? -1.f : sdf > 0.f); // determine threshold, 1.f currently
    }
    else return -1.f; // return - of threshold
}
//
////λ = ||K^-1*x||2
__device__ float calculateCurrentTSDF(Matrix4f pose_traj, Matrix4f depth_ext,
                                      Matrix4f m_traj, Matrix3f depth_intrinsics, float depth,
                                      Matrix3f intrinsics, Vector3f p, float truncation_distance){
    float current_tsdf = -1.f * ((1.f / calculateLambda(depth_ext, m_traj, depth_intrinsics, intrinsics, p)) *
            (pose_traj.block<3, 1>(0, 3) - p).norm() - depth);
    return calculateSDF_truncation(truncation_distance, current_tsdf);
}
//
//
//// calculate weighted running tsdf average
__device__ float calculateWeightedTSDF(int current_weight, float current_tsdf, int new_weight, float new_tsdf){
    float updated_tsdf = (current_weight * current_tsdf + new_weight * new_tsdf) /
                         (current_weight + new_weight);
    return updated_tsdf;
}
//
//// calculate weighted running weight average
__device__ int calculateWeightedAvgWeight(int current_weight, int new_weight){
    return current_weight + new_weight;
}
//
//// truncate updated weight
__device__ int calculateTruncatedWeight(int weighted_avg, int some_value){
    if(weighted_avg < some_value)
        return weighted_avg;
    return some_value;
//    return std::min(weighted_avg, some_value);
}
//
__device__ Vector4uc calculateWeightedColorUpdate(int current_weight, Vector4uc curr_color,
                                                  int new_weight, Vector4uc new_color)
{
    return Vector4uc((current_weight * curr_color[0] + new_weight * new_color[0]) /
                     (current_weight + new_weight),
                     (current_weight * curr_color[1] + new_weight * new_color[1]) /
                     (current_weight + new_weight),
                     (current_weight * curr_color[2] + new_weight * new_color[2]) /
                     (current_weight + new_weight),
                     (current_weight * curr_color[3] + new_weight * new_color[3]) /
                     (current_weight + new_weight));
}

__global__ void updateSurfaceReconstructionGlobal(ImageConstants*& imageConstants,
                                 cv::cuda::PtrStepSz<float> tsdf_values,
                                 cv::cuda::PtrStepSz<float> tsdf_weight,
                                 cv::cuda::PtrStepSz<Vector4uc> tsdf_color,
                                 cv::cuda::PtrStepSz<Vector4uc> color_map,
                                 cv::cuda::PtrStepSz<float> depth_map,
                                 int width, int height,
                                 float voxel_scale,
                                 int volume_size,
                                 Matrix4f depth_ext,
                                 Matrix4f m_traj,
                                 Matrix3f depth_intrinsics,
                                 Matrix3f depth_intrinsics_inv,
                                 Matrix4f pose_traj,
                                 float truncation_distance){

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    int truncate_updated_weight = 128;

    for(int k=0; k < volume_size; ++k) {
        const Vector3f global_coord((static_cast<float>(threadX) + 0.5f) * voxel_scale,
                               (static_cast<float>(threadY) + 0.5f) * voxel_scale,
                               (static_cast<float>(k) + 0.5f) * voxel_scale);

        Vector3f camera_coord = (depth_ext * m_traj *
                                 Vector4f(global_coord.x(),
                                          global_coord.y(), global_coord.z(), 1.0f)).block<3, 1>(0, 0);

        if (camera_coord.z() <= 0) continue;

        Vector2i image_coord = perspective_projection(depth_ext, m_traj, depth_intrinsics, global_coord);

        if (image_coord.x() < 0 || image_coord.x() >= width
            || image_coord.y() < 0 || image_coord.y() >= height)
            continue;

        float depth = depth_map.ptr((int) image_coord.y())[image_coord.x()];

        if (depth <= 0) continue;

        float F_rk = calculateCurrentTSDF(pose_traj, depth_ext, m_traj, depth_intrinsics, depth,
                                          depth_intrinsics_inv, global_coord, truncation_distance);

        printf("3333\n");

        if (F_rk == -1.f) continue;
        printf("4444\n");

        int W_k = 1;
        // TODO: it should be y, if y!=z, change it volume_y
        int prev_weight = tsdf_weight.ptr(k * volume_size + threadY)[threadX];
        int prev_tsdf = tsdf_values.ptr(k * volume_size + threadY)[threadX];

        float updated_tsdf = calculateWeightedTSDF(prev_weight, prev_tsdf, W_k, F_rk);

        printf("5555\n");

        int updated_W_k = calculateWeightedAvgWeight(prev_weight, W_k);
        int truncated_weight = calculateTruncatedWeight(updated_W_k, truncate_updated_weight);

        // TODO: it should be y, if y!=z, change it volume_y
        tsdf_values.ptr(k * volume_size + threadY)[threadX] = updated_tsdf;
        tsdf_weight.ptr(k * volume_size + threadY)[threadX] = truncated_weight;

        printf("7777\n");

        Vector4uc curr_color;
        if (F_rk <= truncation_distance / 2 &&
            F_rk >= -truncation_distance / 2) {
            // TODO: it should be y, if y!=z, change it volume_y
            Vector4uc prev_color = tsdf_color.ptr(k * volume_size + threadY)[threadX];
            Vector4uc image_color = color_map.ptr(image_coord.y())[image_coord.x()];
            curr_color = calculateWeightedColorUpdate(prev_weight, prev_color,
                                                      truncated_weight, image_color);
            // TODO: it should be y, if y!=z, change it volume_y
            tsdf_color.ptr(k * volume_size + threadY)[threadX] = curr_color;

        }

      }
}
void updateSurfaceReconstruction(Pose* pose,ImageConstants* imageConstants,
                                                  ImageData* imageData, SurfaceLevelData* surf_data,  GlobalVolume* global_volume)
{
//    printf("1234\n");

    const dim3 threads(8, 8);
    const dim3 blocks((global_volume->volume_size.x + threads.x - 1) / threads.x,
                      (global_volume->volume_size.y + threads.y - 1) / threads.y);
    cv::cuda::GpuMat& tsdf_vals = global_volume->TSDF_values;
    cv::cuda::GpuMat& tsdf_weights = global_volume->TSDF_weight;
    cv::cuda::GpuMat& tsdf_color = global_volume->TSDF_color;
    cv::cuda::GpuMat& color_map = imageData->m_colorMap;
    cv::cuda::GpuMat& depth_map = imageData->m_depthMap;
//    printf("5678\n");

    updateSurfaceReconstructionGlobal<<<blocks,threads>>>(imageConstants,
                                     tsdf_vals,tsdf_weights, tsdf_color,
                                     color_map, depth_map,
                                     imageConstants->m_colorImageWidth,
                                     imageConstants->m_colorImageHeight,
                                     global_volume->voxel_scale,
                                     global_volume->volume_size.z,
                                     imageConstants->m_depthExtrinsics,
                                     imageConstants->m_trajectory,
                                     imageConstants->m_depthIntrinsics,
                                     imageConstants->m_depthIntrinsicsInv,
                                     pose->m_trajectory,
                                     global_volume->truncation_distance);
    // debugging purposes.
    printf("7890\n");
    cv::Mat result;
    tsdf_vals.download(result);
    std::cout << result.at<float>(0, 0) << std::endl;
    assert(cudaSuccess == cudaDeviceSynchronize());

}

