//
// Created by ilteber on 05.07.21.
//
#include "data_types.h"

#include "common_functions.h"
#include "surface_reconstruction.h"

//HELPER FUNCTIONS
__device__ float calculateLambda( ImageConstants*& imageConstants, Matrix3f intrinsicsInv, Vector3f p){
    Vector2i projected = perspective_projection(imageConstants,p);
    Vector3f dot_p = Vector3f(projected.x(), projected.y(), 1.0f);
    return (intrinsicsInv * dot_p).norm();
}

__device__ Vector2i perspective_projection(ImageConstants*& imageConstants, Vector3f p)
{
    Vector4f p_temp = Vector4f(p.x(), p.y(), p.z(), 1.0);
    Matrix4f identity = Matrix4f::Zero();
    identity.block<3, 3>(0, 0) = Matrix3f::Identity();
    Vector3f p2 = imageConstants->m_depthIntrinsics * identity.block<3, 4>(0, 0) * imageConstants->m_depthExtrinsics * imageConstants->m_trajectory * p_temp;
    return Vector2i((int) round(p2.x() / p2.z()), (int) round(p2.y() / p2.z()));
}


__device__ float calculateSDF_truncation(float truncation_distance, float sdf){
    if (sdf >= -truncation_distance) {
        return fmin(1.f, sdf / truncation_distance) * (sdf < 0 ? -1 : sdf > 0); // determine threshold, 1.f currently
    }
    else return -1.f; // return - of threshold
}

//λ = ||K^-1*x||2
__device__ float calculateCurrentTSDF(ImageConstants*& imageConstants, float depth, Matrix3f intrinsics, Vector3f p, float truncation_distance){
    float current_tsdf = (1.f / calculateLambda(imageConstants,intrinsics, p)) * (get_translation(imageConstants) - p).norm() - depth;
    return calculateSDF_truncation(truncation_distance, current_tsdf);
}


// calculate weighted running tsdf average
__device__ float calculateWeightedTSDF(int current_weight, float current_tsdf, int new_weight, float new_tsdf){
    float updated_tsdf = (current_weight * current_tsdf + new_weight * new_tsdf) /
                         (current_weight + new_weight);
    return updated_tsdf;
}

// calculate weighted running weight average
__device__ int calculateWeightedAvgWeight(int current_weight, int new_weight){
    return current_weight + new_weight;
}

// truncate updated weight
__device__ int calculateTruncatedWeight(int weighted_avg, int some_value){
    return std::min(weighted_avg, some_value);
}

__device__ Vector4uc calculateWeightedColorUpdate(int current_weight, Vector4uc curr_color, int new_weight, Vector4uc new_color)
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

__global__ void updateSurfaceReconstructionGlobal(Pose* pose,ImageConstants*& imageConstants,
                                 ImageData* imageData, SurfaceLevelData* surf_data,GlobalVolume* global_volume,
                                 cv::cuda::PtrStepSz<int> tsdf_values,cv::cuda::PtrStepSz<int> tsdf_weight,cv::cuda::PtrStepSz<unsigned int > tsdf_color,
                                 cv::cuda::PtrStepSz<float> depth_map, int width, int height){

    int threadX = blockIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = blockIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    int truncate_updated_weight = 128; // check the intuition!
//    this->imageProperties = image_properties;
            for(int k=0; k < global_volume->volume_size.z; k++) {

                Vector3f global_coord(threadX, threadY, k);

//                if (!global_coord.allFinite()) continue;

                //Vector3f camera_coord = get_rotation(image_properties) * global_coord + get_translation(image_properties);
                Vector3f camera_coord = (imageConstants->m_depthExtrinsics * imageConstants->m_trajectory *
                                         Vector4f(global_coord.x(),
                                                  global_coord.y(), global_coord.z(), 1.0f)).block<3, 1>(0, 0);
                if (camera_coord.z() <= 0) continue;

                Vector2i image_coord = perspective_projection(imageConstants,global_coord); // check the calculation is true!!

                if (image_coord.x() < 0 || image_coord.x() >= width
                    || image_coord.y() < 0 || image_coord.y() >= height)
                    continue;

                int index = image_coord.x() + image_coord.y() * width;
                float depth = depth_map.ptr((int) image_coord.y())[image_coord.x()];
//                float depth = _curr_lvl_data((int) image_coord.y())[(int) image_coord.x()];

                if (depth == MINF || depth <= 0) continue;

                float F_rk = calculateCurrentTSDF(imageConstants, depth, imageConstants->m_depthIntrinsicsInv,
                                                  global_coord, global_volume->truncation_distance);

                if (F_rk == -1.f) continue;

                int W_k = 1;
                int prev_weight = tsdf_weight.ptr(k * global_volume->volume_size.y + threadY)[threadX];
                int prev_tsdf = tsdf_values.ptr(k * global_volume->volume_size.y + threadY)[threadX];
                float updated_tsdf = calculateWeightedTSDF(prev_weight, prev_tsdf, W_k,
                                                           F_rk);

                int truncated_weight = calculateTruncatedWeight(calculateWeightedAvgWeight
                                                                        (tsdf_weight.ptr(k * global_volume->volume_size.y + threadY)[threadX], W_k),
                                                                truncate_updated_weight);

                /*std::cout << "i: " << i << ", j: " << j << ", k: " << k << std::endl;
                std::cout << "depth: " << depth << std::endl;
                std::cout << "F_rk: " << F_rk << std::endl;
                std::cout << "updated_tsdf: " << updated_tsdf << std::endl;
                std::cout << "truncated_weight: " << truncated_weight << std::endl << std::endl;*/


                tsdf_values.ptr(k * global_volume->volume_size.y + threadY)[threadX] = (int) updated_tsdf;
                tsdf_weight.ptr(k * global_volume->volume_size.y + threadY)[threadX] = (int) truncated_weight;

//                Voxel curr_voxel;
//                curr_voxel.tsdf_distance_value = updated_tsdf;
//                curr_voxel.tsdf_weight = truncated_weight;

                Vector4uc curr_color;
                if (F_rk <= global_volume->truncation_distance / 2 &&
                    F_rk >= -global_volume->truncation_distance / 2) {
                    // TODO: check here!!
                    Vector4uc prev_color = global_volume->get(i, j, k).color;
                    curr_color = Vector4uc(imageData->m_colorMap[index],
                                           imageData->m_colorMap[index + 1],
                                           imageData->m_colorMap[index + 2],
                                           imageData->m_colorMap[index + 3]);
                    curr_color = calculateWeightedColorUpdate(prev_weight, prev_color, W_k, curr_color);
                    curr_voxel.color = curr_color;
                }

//                global_volume->set(i, j, k, curr_voxel);  // check whether assign is successful
            }

//    image_properties = this->imageProperties;
}
void updateSurfaceReconstruction(Pose* pose,ImageConstants*& imageConstants,
                                                  ImageData* imageData, SurfaceLevelData* surf_data, GlobalVolume* global_volume)
{
    const dim3 threads(32, 32);
    const dim3 blocks((global_volume->volume_size.x + threads.x - 1) / threads.x,
                      (global_volume->volume_size.y + threads.y - 1) / threads.y);

    updateSurfaceReconstructionGlobal<<<blocks,threads>>>(pose,imageConstants,
                                     imageData,surf_data,global_volume,
                                     global_volume->TSDF_values,global_volume->TSDF_weight,global_volume->TSDF_color,
                                     surf_data->curr_level_data[0],surf_data->level_img_width[0],surf_data->level_img_height[0]);

    assert(cudaSuccess == cudaDeviceSynchronize());

}


//private:
//
//float depth_margin;                 //μ
//float *m_depthMap;                  //Rk
//ImageProperties* imageProperties;
