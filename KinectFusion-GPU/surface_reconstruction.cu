//
// Created by ilteber on 05.07.21.
//
#include "data_types.h"

#include "common_functions.h"
#include "surface_reconstruction.h"
#define N 1048576
#define MAX_ERR 1e-6
////HELPER FUNCTIONS
//__device__ float calculateLambda( ImageConstants*& imageConstants, Matrix3f intrinsicsInv, Vector3f p){
//    Vector2i projected = perspective_projection(imageConstants,p);
//    Vector3f dot_p = Vector3f(projected.x(), projected.y(), 1.0f);
//    return (intrinsicsInv * dot_p).norm();
//}
//
//// TODO:: Check for float2int conversion
//__device__ Vector2i perspective_projection(ImageConstants*& imageConstants, Vector3f p)
//{
//    Vector4f p_temp = Vector4f(p.x(), p.y(), p.z(), 1.0);
//    Matrix4f identity = Matrix4f::Zero();
//    identity.block<3, 3>(0, 0) = Matrix3f::Identity();
//    Vector3f p2 = imageConstants->m_depthIntrinsics * identity.block<3, 4>(0, 0) * imageConstants->m_depthExtrinsics * imageConstants->m_trajectory * p_temp;
//    return Vector2i((int) round(p2.x() / p2.z()), (int) round(p2.y() / p2.z()));
//}
//
//
//__device__ float calculateSDF_truncation(float truncation_distance, float sdf){
//    if (sdf >= -truncation_distance) {
//        return fmin(1.f, sdf / truncation_distance) * (sdf < 0.f ? -1.f : sdf > 0.f); // determine threshold, 1.f currently
//    }
//    else return -1.f; // return - of threshold
//}
//
////λ = ||K^-1*x||2
//__device__ float calculateCurrentTSDF(Pose* pose,ImageConstants*& imageConstants, float depth, Matrix3f intrinsics, Vector3f p, float truncation_distance){
//    float current_tsdf = -1.f * ((1.f / calculateLambda(imageConstants,intrinsics, p)) * (pose->m_trajectory.block<3, 1>(0, 3) - p).norm() - depth);
//    return calculateSDF_truncation(truncation_distance, current_tsdf);
//}
//
//
//// calculate weighted running tsdf average
//__device__ float calculateWeightedTSDF(int current_weight, float current_tsdf, int new_weight, float new_tsdf){
//    float updated_tsdf = (current_weight * current_tsdf + new_weight * new_tsdf) /
//                         (current_weight + new_weight);
//    return updated_tsdf;
//}
//
//// calculate weighted running weight average
//__device__ int calculateWeightedAvgWeight(int current_weight, int new_weight){
//    return current_weight + new_weight;
//}
//
//// truncate updated weight
//__device__ int calculateTruncatedWeight(int weighted_avg, int some_value){
//    if(weighted_avg < some_value)
//        return weighted_avg;
//    return some_value;
////    return std::min(weighted_avg, some_value);
//}
//
//__device__ Vector4uc calculateWeightedColorUpdate(int current_weight, Vector4uc curr_color, int new_weight, Vector4uc new_color)
//{
//    return Vector4uc((current_weight * curr_color[0] + new_weight * new_color[0]) /
//                     (current_weight + new_weight),
//                     (current_weight * curr_color[1] + new_weight * new_color[1]) /
//                     (current_weight + new_weight),
//                     (current_weight * curr_color[2] + new_weight * new_color[2]) /
//                     (current_weight + new_weight),
//                     (current_weight * curr_color[3] + new_weight * new_color[3]) /
//                     (current_weight + new_weight));
//}

__global__ void updateSurfaceReconstructionGlobal(float* tavuk,Pose* pose,ImageConstants*& imageConstants,
                                 ImageData* imageData, SurfaceLevelData* surf_data,GlobalVolume*& global_volume,
                                 cv::cuda::PtrStepSz<float> tsdf_values,cv::cuda::PtrStepSz<float> tsdf_weight,cv::cuda::PtrStepSz<Vector4uc> tsdf_color,
                                 cv::cuda::PtrStepSz<Vector4uc> color_map, cv::cuda::PtrStepSz<float> depth_map, int width, int height){

    printf("asdasdsadasd");
    int threadX = blockIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = blockIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;
    int truncate_updated_weight = 128; // check the intuition!
    float voxel_scale = global_volume->voxel_scale;
    for(int k=0; k < global_volume->volume_size.z; ++k) {
        const Vector3f global_coord((static_cast<float>(threadX) + 0.5f) * voxel_scale,
                               (static_cast<float>(threadY) + 0.5f) * voxel_scale,
                               (static_cast<float>(k) + 0.5f) * voxel_scale);
//        Vector3f global_coord(threadX*voxel_scale, threadY*voxel_scale, k*voxel_scale);

        Vector3f camera_coord = (imageConstants->m_depthExtrinsics * imageConstants->m_trajectory *
                                 Vector4f(global_coord.x(),
                                          global_coord.y(), global_coord.z(), 1.0f)).block<3, 1>(0, 0);
        if (camera_coord.z() <= 0) continue;

        Vector4f p_temp2 = Vector4f(global_coord.x(), global_coord.y(), global_coord.z(), 1.0);
        Matrix4f identity2 = Matrix4f::Zero();
        identity2.block<3, 3>(0, 0) = Matrix3f::Identity();
        Vector3f p22 = imageConstants->m_depthIntrinsics * identity2.block<3, 4>(0, 0) * imageConstants->m_depthExtrinsics * imageConstants->m_trajectory * p_temp2;
        Vector2i image_coord = Vector2i((int) round(p22.x() / p22.z()), (int) round(p22.y() / p22.z()));

//        Vector2i image_coord = perspective_projection(imageConstants,global_coord); // check the calculation is true!!

        if (image_coord.x() < 0 || image_coord.x() >= width
            || image_coord.y() < 0 || image_coord.y() >= height)
            continue;

        int index = image_coord.x() + image_coord.y() * width;


        float depth = depth_map.ptr((int) image_coord.y())[image_coord.x()];

        if (depth <= 0) continue;

        float F_rk;
//        = calculateCurrentTSDF(pose,imageConstants, depth, imageConstants->m_depthIntrinsicsInv,
//                                          global_coord, global_volume->truncation_distance);

        //perspective_projection
        Vector4f p_temp = Vector4f(global_coord.x(), global_coord.y(), global_coord.z(), 1.0);
        Matrix4f identity = Matrix4f::Zero();
        identity.block<3, 3>(0, 0) = Matrix3f::Identity();
        Vector3f p2 = imageConstants->m_depthIntrinsics * identity.block<3, 4>(0, 0) * imageConstants->m_depthExtrinsics * imageConstants->m_trajectory * p_temp;
        Vector2i projected = Vector2i((int) round(p2.x() / p2.z()), (int) round(p2.y() / p2.z()));
        // Calculate lambda
        Vector3f dot_p = Vector3f(projected.x(), projected.y(), 1.0f);
        float lambda =  (imageConstants->m_depthIntrinsicsInv * dot_p).norm();
        //calculateCurrentTSDF
        float current_tsdf = -1.f * ((1.f / lambda) * (pose->m_trajectory.block<3, 1>(0, 3) - global_coord).norm() - depth);
        //calculateSDF_truncation
        if (current_tsdf >= -global_volume->truncation_distance) {
            F_rk = fmin(1.f, current_tsdf / global_volume->truncation_distance) * (current_tsdf < 0.f ? -1.f : current_tsdf > 0.f); // determine threshold, 1.f currently
        }
        else
        {
            F_rk = -1.f; // return - of threshold
        }

        if (F_rk == -1.f) continue;

        int W_k = 1;
        int prev_weight = tsdf_weight.ptr(k * global_volume->volume_size.y + threadY)[threadX];
        int prev_tsdf = tsdf_values.ptr(k * global_volume->volume_size.y + threadY)[threadX];
        // calculateWeightedTSDF
        float updated_tsdf = (prev_weight * prev_tsdf + W_k * F_rk) /
                             (prev_weight + W_k);
        float truncated_weight = 0;
        //calculateTruncatedWeight
        if(prev_weight + W_k < truncate_updated_weight)
            truncated_weight = prev_weight + W_k;
        else
        {
            truncated_weight = truncate_updated_weight;
        }
        *tavuk = 43.f;

//                std::cout << "i: " << i << ", j: " << j << ", k: " << k << std::endl;
//                std::cout << "depth: " << depth << std::endl;
//                std::cout << "F_rk: " << F_rk << std::endl;
//                std::cout << "updated_tsdf: " << updated_tsdf << std::endl;
//                std::cout << "truncated_weight: " << truncated_weight << std::endl << std::endl;
//        printf("depth: %f" , depth);
//        printf("F_rk: %f" , F_rk);
//        printf("updated_tsdf: %f" , updated_tsdf);
//        printf("truncated_weight: %f" , truncated_weight);

        tsdf_values.ptr(k * global_volume->volume_size.y + threadY)[threadX] = updated_tsdf;
        tsdf_weight.ptr(k * global_volume->volume_size.y + threadY)[threadX] = truncated_weight;
//        tsdf_values.ptr(k * global_volume->volume_size.y + threadY)[threadX] = 2;
//        tsdf_weight.ptr(k * global_volume->volume_size.y + threadY)[threadX] = 3;
//                Voxel curr_voxel;
//                curr_voxel.tsdf_distance_value = updated_tsdf;
//                curr_voxel.tsdf_weight = truncated_weight;

        Vector4uc curr_color;
        if (F_rk <= global_volume->truncation_distance / 2 &&
            F_rk >= -global_volume->truncation_distance / 2) {
            // TODO: check here!!
            Vector4uc prev_color = tsdf_color.ptr(k * global_volume->volume_size.y + threadY)[threadX];
            Vector4uc image_color = color_map.ptr(image_coord.y())[image_coord.x()];
            //calculateWeightedColorUpdate
            curr_color = Vector4uc((prev_weight * prev_color[0] + W_k * image_color[0]) /
                             (prev_weight + W_k),
                             (prev_weight * prev_color[1] + W_k * image_color[1]) /
                             (prev_weight + W_k),
                             (prev_weight * prev_color[2] + W_k * image_color[2]) /
                             (prev_weight + W_k),
                             (prev_weight * prev_color[3] + W_k * image_color[3]) /
                             (prev_weight + W_k));
//            curr_color = calculateWeightedColorUpdate(prev_weight, prev_color, W_k, image_color);
//                    cur_color.block<3, 1>(0,0) = {0,1,2,3};
            tsdf_color.ptr(k * global_volume->volume_size.y + threadY)[threadX] = curr_color;

        }

      }

//    image_properties = this->imageProperties;
}
void updateSurfaceReconstruction(Pose* pose,ImageConstants* imageConstants,
                                                  ImageData* imageData, SurfaceLevelData* surf_data, GlobalVolume* global_volume)
{

    float x = 5;
    float* tavuk;
    cudaMalloc(&tavuk, sizeof(float));
    cudaMemcpy(&tavuk, &x, sizeof(float),cudaMemcpyHostToDevice);

    const dim3 threads(32, 32);
    const dim3 blocks((global_volume->volume_size.x + threads.x - 1) / threads.x,
                      (global_volume->volume_size.y + threads.y - 1) / threads.y);
    cv::cuda::GpuMat& tsdf_vals = global_volume->TSDF_values;
    cv::cuda::GpuMat& tsdf_weights = global_volume->TSDF_weight;
    std::cout << "TAVUK GIRIYO : " << tavuk << std::endl;
    std::cout << "TAVUK GIRIYO : " << *tavuk << std::endl;
    updateSurfaceReconstructionGlobal<<<blocks,threads>>>(tavuk,pose,imageConstants,
                                     imageData,surf_data,global_volume,
                                                          tsdf_vals,tsdf_weights,global_volume->TSDF_color,
                                    imageData->m_colorMap,imageData->m_depthMap,imageConstants->m_colorImageWidth,imageConstants->m_colorImageHeight);
    std::cout << "TAVUK GELDI : " << tavuk << std::endl;
    std::cout << "TAVUK GELDI : " << *tavuk << std::endl;
//    cv::Mat result;
//    tsdf_vals.download(result);
//    std::cout << result;
    assert(cudaSuccess == cudaDeviceSynchronize());

}

