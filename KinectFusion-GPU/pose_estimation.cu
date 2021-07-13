//
// Created by Beyza Tugce Bilgic on 06.07.21.
//

#include "pose_estimation.h"

//std::vector<cv::cuda::GpuMat> global_vertex_map;
//std::vector<cv::cuda::GpuMat> global_normal_map;

// The distance threshold in mm
float distance_threshold { 10.f };
// The angle threshold in degrees
float angle_threshold { 20.f };

void pose_estimate(const std::vector<int>&  iterations, ImageConstants*& imageConstants, ImageData* imageData, SurfaceLevelData* surf_data){

    int level = surf_data->level - 1;

    for ( int i = level; i >= 0; i--) {

        dim3 block(8, 8);

        float rows = surf_data->level_img_width[i];
        float cols = surf_data->level_img_height[i];

        cv::cuda::GpuMat& vertex_map = surf_data->vertex_map[i];
        cv::cuda::GpuMat& normal_map = surf_data->normal_map[i];
        cv::cuda::GpuMat& vertex_map_predicted = surf_data->vertex_map_predicted[i];
        cv::cuda::GpuMat& normal_map_predicted = surf_data->normal_map_predicted[i];

        dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

        int width = surf_data->level_img_width[i];
        int height = surf_data->level_img_height[i];

        Matrix3f rotation = imageConstants->m_trajectory.block<3, 3>(0, 0);
        Vector3f translation = imageConstants->m_trajectory.block<3, 1>(0, 3);

        init_global_map(imageData->m_depthMap, vertex_map, normal_map, rotation, translation, width, height);

        pose_estimate_helper<<<grid, block>>>(iterations, i, imageData->m_depthMap, vertex_map, vertex_map_predicted, normal_map_predicted, rotation, translation);

    }
}

__global__ void pose_estimate_helper(std::vector<int> iterations, int level,
                                     cv::cuda::PtrStepSz<float> depth_map,
                                     cv::cuda::PtrStepSz<Vector3f> vertex_map,
                                     cv::cuda::PtrStepSz<Vector3f> vertex_map_predicted,
                                     cv::cuda::PtrStepSz<Vector3f> normal_map_predicted,
                                     Matrix3f &rotation, Vector3f &translation){

    Isometry3f T;

    for ( int j = 0; j < iterations[level]; j++) {
            T = point_to_plane( global_vertex_map, vertex_map_predicted, normal_map_predicted, level, T);
            // Return the new pose
            rotation = T.rotation();
            translation = T.translation();
    }
}


//https://www.cs.princeton.edu/~smr/papers/icpstability.pdf
//https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf
__global__ void point_to_plane( std::vector<Vector3f> source, std::vector<Vector3f> dest, std::vector<Vector3f> normal, int level, Isometry3f& T){

    bool validity_check = false;
    Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A {};
    Eigen::Matrix<float, 6, 1> b {};

    for ( int i = 0; i < source.size(); i++){
        dim3 block(8, 8);
        float rows = surf_data->level_img_width[i];
        float cols = surf_data->level_img_height[i];
        dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        check_correspondence_validity<<grid, block>>(level, i, vertex_map_predicted, normal_map_predicted, validity_check);
        if (validity_check){
            Vector3f pointToNormal = source[i].cross(normal[i]);
            A.block<3,3>(0,0) += pointToNormal * pointToNormal.transpose();
            A.block<3,3>(0,3) += normal[i] * pointToNormal.transpose();
            A.block<3,3>(3,3) += normal[i] * normal[i].transpose();
            A.block<3,3>(3,0) = A.block<3,3>(0,3);

            float sum = (source[i] - dest[i]).dot(normal[i]);
            b.head(3) -= pointToNormal * sum;
            b.tail(3) -= normal[i] * sum;
        }
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
__global__ void check_correspondence_validity(int level, int point, cv::cuda::PtrStepSz<Vector3f> vertex_map_predicted, cv::cuda::PtrStepSz<Vector3f> normal_map_predicted, bool& validity){
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
    float distance = (vertex_map_predicted.ptr((point-1).y())[(point-1).x()] - global_vertex_map.ptr((point-1).y())[(point-1).x()]).norm();

    //const float sine = normal_current_global.cross(normal_previous_global).norm();
    float angle = global_normal_map.ptr((point-1).y())[(point-1).x()].
            cross(normal_map_predicted[point-1]).norm();

    if ( currDepthValue != MINF && !isnan(currDepthValue) && distance <= distance_threshold && angle <= angle_threshold ){
        validity = true;
    }
    validity = false;
}


//Get current global vertex map from surface measurement
__global__ void get_global_vertex_map( cv::cuda::PtrStepSz<float> depth_map, cv::cuda::PtrStep<Vector3f> vertex_map,
                                       Matrix3f rotation, Vector3f translation, int width, int height) {

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
__global__ void get_global_normal_map( cv::cuda::PtrStepSz<float> depth_map, cv::cuda::PtrStep<Vector3f> normal_map, cv::cuda::PtrStep<Vector3f> &global_normal_map,
                                      Matrix3f rotation, Vector3f translation, int width, int height, int level){

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

void init_global_map(cv::cuda::PtrStepSz<float> depth_map, cv::cuda::PtrStep<Vector3f> vertex_map, cv::cuda::PtrStep<Vector3f> normal_map,
                     Matrix3f rotation, Vector3f translation, int width, int height) {

    dim3 block(8, 8);
    float rows = width;
    float cols = height;
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    global_vertex_map.push_back(cv::cuda::createContinuous(width, height, CV_32FC3));
    get_global_vertex_map<<grid, block>>( depth_map, vertex_map, global_vertex_map, rotation, translation, width, height);

    global_normal_map.push_back(cv::cuda::createContinuous(width, height, CV_32FC3));
    get_global_normal_map<<grid, block>>( depth_map, normal_map, global_normal_map, rotation, translation, width, height);

}
