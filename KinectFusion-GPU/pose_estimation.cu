//
// Created by Beyza Tugce Bilgic on 06.07.21.
//

#include "pose_estimation.h"

//std::vector<cv::cuda::GpuMat> global_vertex_map;
//std::vector<cv::cuda::GpuMat> global_normal_map;



void pose_estimate(const std::vector<int>&  iterations,
                   ImageConstants*& imageConstants,
                   ImageData* imageData, SurfaceLevelData* surf_data){

    int level = surf_data->level - 1;

    std::vector<cv::cuda::GpuMat> global_vertex_maps;
    std::vector<cv::cuda::GpuMat> global_normal_maps;

    for ( int i = level; i >= 0; i--) {

        int width = surf_data->level_img_width[i];
        int height = surf_data->level_img_height[i];

        float rows = height;
        float cols = width;

        global_vertex_maps.push_back(cv::cuda::createContinuous(width, height, CV_32FC3));
        global_normal_maps.push_back(cv::cuda::createContinuous(width, height, CV_32FC3));

        dim3 block(8, 8);


        cv::cuda::GpuMat& vertex_map = surf_data->vertex_map[i];
        cv::cuda::GpuMat& normal_map = surf_data->normal_map[i];
        cv::cuda::GpuMat& vertex_map_predicted = surf_data->vertex_map_predicted[i];
        cv::cuda::GpuMat& normal_map_predicted = surf_data->normal_map_predicted[i];
        cv::cuda::GpuMat& global_vertex_map = global_vertex_maps[i];
        cv::cuda::GpuMat& global_normal_map = global_normal_maps[i];

        dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

        Matrix3f rotation = imageConstants->m_trajectory.block<3, 3>(0, 0);
        Vector3f translation = imageConstants->m_trajectory.block<3, 1>(0, 3);

        init_global_map(imageData->m_depthMap, vertex_map, normal_map,global_vertex_map,global_normal_map,
                        rotation, translation, width, height, level);

        pose_estimate_helper(iterations[i], imageData->m_depthMap, vertex_map, vertex_map_predicted,
                             normal_map_predicted, global_vertex_map, global_normal_map, width, height,
                             rotation, translation);

    }
}

void pose_estimate_helper( int iteration,
                           cv::cuda::PtrStepSz<float> depth_map,
                           cv::cuda::PtrStepSz<Vector3f> vertex_map,
                           cv::cuda::PtrStepSz<Vector3f> vertex_map_predicted,
                           cv::cuda::PtrStepSz<Vector3f> normal_map_predicted,
                           cv::cuda::PtrStepSz<Vector3f> global_vertex_map,
                           cv::cuda::PtrStepSz<Vector3f> global_normal_map,
                           int width, int height,
                           Matrix3f &rotation, Vector3f &translation){

    Isometry3f T;

    for ( int j = 0; j < iteration; j++) {
            point_to_plane( global_vertex_map, vertex_map_predicted, normal_map_predicted,
                            global_normal_map,depth_map,
                            width, height, T);
            // Return the new pose
            rotation = T.rotation();
            translation = T.translation();
    }
}


//https://www.cs.princeton.edu/~smr/papers/icpstability.pdf
//https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf
void point_to_plane( cv::cuda::PtrStepSz<Vector3f> source,
                     cv::cuda::PtrStepSz<Vector3f> dest,
                     cv::cuda::PtrStepSz<Vector3f> normal,
                     cv::cuda::PtrStepSz<Vector3f> global_normal_map,
                     cv::cuda::PtrStepSz<float> depth_map,
                     int width, int height,
                     Isometry3f& T){

    bool validity_check = false;
    Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A {};
    Eigen::Matrix<float, 6, 1> b {};

    float rows = height;
    float cols = width;

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= rows or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= cols or threadY < 0)
        return;

    dim3 block(8, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    //Vector3f point = source.ptr(threadY)[threadX];
    check_correspondence_validity<<<grid, block>>>(width,height,
                                                   dest, normal, source, global_normal_map, depth_map, validity_check);
    if (validity_check){
        Vector3f pointToNormal = source.ptr(threadY)[threadX].cross(normal.ptr(threadY)[threadX]);
        A.block<3,3>(0,0) += pointToNormal * pointToNormal.transpose();
        A.block<3,3>(0,3) += normal.ptr(threadY)[threadX] * pointToNormal.transpose();
        A.block<3,3>(3,3) += normal.ptr(threadY)[threadX] * normal.ptr(threadY)[threadX].transpose();
        A.block<3,3>(3,0) = A.block<3,3>(0,3);

        float sum = (source.ptr(threadY)[threadX] - dest.ptr(threadY)[threadX]).dot(normal.ptr(threadY)[threadX]);
        b.head(3) -= pointToNormal * sum;
        b.tail(3) -= normal.ptr(threadY)[threadX] * sum;
    }

    Matrix<float,6,1> x = A.ldlt().solve(b);
    T = Isometry3f::Identity();
    T.linear() = ( AngleAxisf(x(0), Vector3f::UnitX())
                   * AngleAxisf(x(1), Vector3f::UnitY())
                   * AngleAxisf(x(2), Vector3f::UnitZ())
    ).toRotationMatrix();
    T.translation() = x.block(3,0,3,1);
}




//Check correspondences on the distance of vertices and difference in normal values
__global__ void check_correspondence_validity(int width, int height,
                                              cv::cuda::PtrStepSz<Vector3f> vertex_map_predicted,
                                              cv::cuda::PtrStepSz<Vector3f> normal_map_predicted,
                                              cv::cuda::PtrStepSz<Vector3f> global_vertex_map,
                                              cv::cuda::PtrStepSz<Vector3f> global_normal_map,
                                              cv::cuda::PtrStepSz<float> depth_map,
                                              bool& validity){
    /*if ( global_vertex_map.size() == 0){
        global_vertex_map.push_back(cv::cuda::createContinuous(width, height, CV_32FC3));
        get_global_vertex_map( depth_map, vertex_map, global_vertex_map, rotation, translation, width, height);
    }
    if ( global_normal_map.size() == 0){
        global_normal_map.push_back(cv::cuda::createContinuous(width, height, CV_32FC3));
        get_global_normal_map( depth_map, normal_map, global_normal_map, rotation, translation, width, height);
    }*/

    //TODO: How to take the depth value of a specific point ??
    float depth_threshold = 100.f;
    float distance_threshold = 10.f;
    float angle_threshold = 20.f;

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    float currDepthValue =  depth_map.ptr(threadY)[threadX];

    if(currDepthValue > depth_threshold | currDepthValue < 0){
        validity = false;
    }

    //TODO: Call get_global_vertex_map and get_global_normal_map once !!!
    float distance = (vertex_map_predicted.ptr(threadY)[threadX] - global_vertex_map.ptr(threadY)[threadX]).norm();

    //const float sine = normal_current_global.cross(normal_previous_global).norm();
    float angle = global_normal_map.ptr(threadY)[threadX].cross(normal_map_predicted.ptr(threadY)[threadX]).norm();

    if ( currDepthValue != MINF && !isnan(currDepthValue) && distance <= distance_threshold && angle <= angle_threshold ){
        validity = true;
    }
    validity = false;
}


//Get current global vertex map from surface measurement
__global__ void get_global_vertex_map( cv::cuda::PtrStepSz<float> depth_map,
                                       cv::cuda::PtrStep<Vector3f> vertex_map,
                                       Matrix3f rotation, Vector3f translation,
                                       int width, int height,
                                       cv::cuda::PtrStep<Vector3f>& global_vertex_map) {

    float depth_threshold = 100.f;

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    float currDepthValue =  depth_map.ptr(threadY)[threadX];

    if(currDepthValue > depth_threshold | currDepthValue < 0){
        global_vertex_map.ptr(threadY)[threadX] = Vector3f(MINF, MINF, MINF);
    }
    else {
        global_vertex_map.ptr(threadY)[threadX] = rotation * vertex_map.ptr(threadY)[threadX] + translation;
    }
}

//Get current global normal map from surface measurement
__global__ void get_global_normal_map( cv::cuda::PtrStepSz<float> depth_map,
                                       cv::cuda::PtrStep<Vector3f> normal_map,
                                       Matrix3f rotation, Vector3f translation,
                                       int width, int height,
                                       cv::cuda::PtrStep<Vector3f> &global_normal_map){

    float depth_threshold = 100.f;

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    float currDepthValue =  depth_map.ptr(threadY)[threadX];

    if(currDepthValue > depth_threshold | currDepthValue < 0){
        global_normal_map.ptr(threadY)[threadX] = Vector3f(MINF, MINF, MINF);
    }
    else {
        global_normal_map.ptr(threadY)[threadX] = rotation * normal_map.ptr(threadY)[threadX] + translation;
    }
}

void init_global_map(cv::cuda::PtrStepSz<float> depth_map,
                     cv::cuda::PtrStep<Vector3f> vertex_map,
                     cv::cuda::PtrStep<Vector3f> normal_map,
                     cv::cuda::PtrStep<Vector3f>& global_vertex_map,
                     cv::cuda::PtrStep<Vector3f>& global_normal_map,
                     Matrix3f rotation, Vector3f translation,
                     int width, int height, int level) {

    dim3 block(8, 8);
    float rows = height;
    float cols = width;

    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

//    global_vertex_map.ptr(threadY)[threadX] = Vector3f(0.f, 0.f, 0.f);
//    global_vertex_map.push_back(cv::cuda::createContinuous(width, height, CV_32FC3));
    get_global_vertex_map<<<grid, block>>>( depth_map, vertex_map, rotation,
            translation, width, height,global_vertex_map);

//    global_normal_map.ptr(threadY)[threadX] = Vector3f(0.f, 0.f, 0.f);
    get_global_normal_map<<<grid, block>>>( depth_map, normal_map, rotation,
                                            translation, width, height,global_normal_map);

}
