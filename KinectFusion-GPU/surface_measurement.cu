//
// Created by eralpkocas on 25.06.21.
//
#include "data_types.h"
#include "surface_measurement.h"

__global__ void helper_compute_vertex_map(SurfaceLevelData* surf_data, ImageConstants img_constants,
                                          cv::cuda::PtrStepSz<float> depth_map,
                                          cv::cuda::PtrStep<Vector3f> vertex_map,
                                          float fX, float fY,
                                          float cX, float cY, int width, int height, int level)
{
    float depth_threshold = 1000.f;

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    float currDepthValue =  depth_map.ptr(threadY)[threadX];

    if(currDepthValue > depth_threshold || currDepthValue < 0){
        currDepthValue = -1.f;
    }
    int pixel_x = threadX;
    int pixel_y = threadY;
    float camera_x = currDepthValue * ((float) pixel_x - cX) / fX;
    float camera_y = currDepthValue * ((float) pixel_y - cY) / fY;
    //Vector4f temp = img_constants.m_trajectoryInv * img_constants.m_depthExtrinsicsInv
     //      * Vector4f(camera_x, camera_y, currDepthValue, 1.f);
    //vertex_map.ptr(threadY)[threadX] = img_constants.m_trajectoryInv * img_constants.m_depthExtrinsicsInv
     //       * Vector3f(camera_x, camera_y, currDepthValue);
     //vertex_map.ptr(threadY)[threadX] = Vector3f(temp.x(), temp.y(), temp.z());
    vertex_map.ptr(threadY)[threadX] = Vector3f(camera_x, camera_y, currDepthValue);
    //TODO: add color if necessary

}


__global__ void helper_compute_normal_map(SurfaceLevelData* surf_data, cv::cuda::PtrStepSz<Vector3f> vertex_map,
                                          cv::cuda::PtrStep<Vector3f> normal_map,
                                          int width, int height, int level)
{

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    Vector3f curr_vertex = vertex_map.ptr(threadY)[threadX];

    if (curr_vertex.z() == 0.f) {
        normal_map.ptr(threadY)[threadX] = Vector3f(0.f, 0.f, 0.f);
        //TODO: add color as 0, 0 ,0 ,0 if necessary??
    } else {
        Vector3f neigh_1 = Vector3f(vertex_map.ptr(threadY - 1)[threadX].x() -
                                    vertex_map.ptr(threadY + 1)[threadX].x(),
                                    vertex_map.ptr(threadY - 1)[threadX].y() -
                                    vertex_map.ptr(threadY + 1)[threadX].y(),
                                    vertex_map.ptr(threadY - 1)[threadX].z() -
                                    vertex_map.ptr(threadY + 1)[threadX].z());

        Vector3f neigh_2 = Vector3f(vertex_map.ptr(threadY)[threadX - 1].x() -
                                    vertex_map.ptr(threadY)[threadX + 1].x(),
                                    vertex_map.ptr(threadY)[threadX - 1].y() -
                                    vertex_map.ptr(threadY)[threadX + 1].y(),
                                    vertex_map.ptr(threadY)[threadX - 1].z() -
                                    vertex_map.ptr(threadY)[threadX + 1].z());

        Vector3f cross_prod = neigh_1.cross(neigh_2);
        cross_prod.normalize();
        if (cross_prod.z() > 0) cross_prod *= -1;
        normal_map.ptr(threadY)[threadX] = cross_prod;
        //TODO: add color
        /*image_properties->camera_reference_points[i].color = Vector4uc(image_properties->m_colorMap[4*i],
                                                                       image_properties->m_colorMap[4*i+1],
                                                                       image_properties->m_colorMap[4*i+2],
                                                                       image_properties->m_colorMap[4*i+3]);*/
    }
}

bool init_multiscale(SurfaceLevelData* surf_data, ImageData img_data)
{
    float bilateral_color_sigma = 1.;
    float bilateral_spatial_sigma = 1.;
    int depth_diameter = 3 * (int) bilateral_color_sigma;

    for(int i=0; i < surf_data->level; i++)
    {
        if(i==0){
            cv::Mat result;
            img_data.m_depthMap.download(result);
            surf_data->curr_level_data[i].upload(result);
            cv::cuda::bilateralFilter(surf_data->curr_level_data[i], surf_data->curr_smoothed_data[i],
                                      depth_diameter, bilateral_color_sigma, bilateral_spatial_sigma, cv::BORDER_DEFAULT);
//            cv::bilateralFilter(surf_data->curr_level_data[i], surf_data->curr_smoothed_data[i],
//                                depth_diameter, bilateral_color_sigma, bilateral_spatial_sigma, cv::BORDER_DEFAULT);
        }
        else{
            cv::cuda::pyrDown(surf_data->curr_smoothed_data[i-1],
                              surf_data->curr_smoothed_data[i]);
//            cv::pyrDown(surf_data->curr_smoothed_data[i-1],
//                              surf_data->curr_smoothed_data[i]);
        }
    }
    return true;
}

void compute_vertex_map(SurfaceLevelData* surf_data, ImageConstants img_constants){
    for(int i=0; i < surf_data->level; i++){
        dim3 block(8, 8);
        float cols = surf_data->level_img_width[i];
        float rows = surf_data->level_img_height[i];
        cv::cuda::GpuMat& depth_map = surf_data->curr_level_data[i];
        cv::cuda::GpuMat& vertex_map = surf_data->vertex_map[i];
        dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        helper_compute_vertex_map<<<grid, block>>>(surf_data, img_constants, depth_map, vertex_map, surf_data->level_fX[i],
                                                   surf_data->level_fY[i], surf_data->level_cX[i], surf_data->level_cY[i],
                                                   surf_data->level_img_width[i], surf_data->level_img_height[i], i);
        assert(cudaSuccess == cudaDeviceSynchronize());
    }
}

void compute_normal_map(SurfaceLevelData* surf_data){
    for(int i=0; i < surf_data->level; i++){
        dim3 block(8, 8);
        float cols = surf_data->level_img_width[i];
        float rows = surf_data->level_img_height[i];
        cv::cuda::GpuMat& vertex_map = surf_data->vertex_map[i];
        cv::cuda::GpuMat& normal_map = surf_data->normal_map[i];
        dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        helper_compute_normal_map<<<grid, block>>>(surf_data, vertex_map, normal_map,
                                                   surf_data->level_img_width[i], surf_data->level_img_height[i], i);
        assert(cudaSuccess == cudaDeviceSynchronize());
    }
}

void surface_measurement_pipeline(SurfaceLevelData* surf_data, ImageData img_data, ImageConstants img_constants){
    init_multiscale(surf_data, img_data);
    compute_vertex_map(surf_data, img_constants);
    compute_normal_map(surf_data);
}