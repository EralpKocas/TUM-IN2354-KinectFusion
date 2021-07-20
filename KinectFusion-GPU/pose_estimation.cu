//
// Created by Beyza Tugce Bilgic on 06.07.21.
//

#include "pose_estimation.h"

//std::vector<cv::cuda::GpuMat> global_vertex_map;
//std::vector<cv::cuda::GpuMat> global_normal_map;

__device__ Vector2i backproject_vertex(const Vector3f& curr_global_vertex,
                                              const Matrix3f& prev_rotation_inv,
                                              const Vector3f& prev_translation,
                                              float fX, float fY)
{
    Vector3f curr_frame_vertex = prev_rotation_inv * (curr_global_vertex - prev_translation);
    Vector2i curr_img_pixel;
    curr_img_pixel.x() = floor(curr_frame_vertex.x() * fX / curr_frame_vertex.z());
    curr_img_pixel.y() = floor(curr_frame_vertex.y() * fY / curr_frame_vertex.z());
    return curr_img_pixel;
}

__device__ bool isVertDistValid(const Vector3f& curr_global_vertex,
                                const Vector3f& prev_global_vertex,
                                float distance_threshold)
{
    //float distance = (vertex_map_predicted.ptr(threadY)[threadX] - global_vertex_map.ptr(threadY)[threadX]).norm();
    float distance = (prev_global_vertex - curr_global_vertex).norm();
    return distance < distance_threshold;
}

__device__ bool isNormAngleValid(const Vector3f& curr_global_normal,
                                 const Vector3f& prev_global_normal,
                                 float angle_threshold)
{
    float angle = curr_global_normal.cross(prev_global_normal).norm();
    return angle < angle_threshold;
}

__device__ void write_linear_eq(MatrixXf& A, VectorXf& b, int i, bool fill_zero,
                                const Vector3f& curr_global_vertex={0, 0, 0},
                                const Vector3f& prev_global_vertex={0, 0, 0},
                                const Vector3f& prev_global_normal={0, 0, 0})
{
    if(fill_zero)
    {
        A(i, 0)=0.f;
        A(i, 1)=0.f;
        A(i, 2)=0.f;
        A(i, 3)=0.f;
        A(i, 4)=0.f;
        A(i, 5)=0.f;
        b(i)=0.f;
    }
    else
    {
        A(i, 0)=1.f;
        A(i, 1)=0.f;
        A(i, 2)=0.f;
        A(i, 3)=0.f;
        A(i, 4)=0.f;
        A(i, 5)=0.f;
        b(i)=0.f;
    }
}

__global__ void form_linear_eq_new(int width, int height,
                               cv::cuda::PtrStepSz<Vector3f> curr_frame_vertex,
                               cv::cuda::PtrStepSz<Vector3f> curr_frame_normal,
                               cv::cuda::PtrStepSz<Vector3f> prev_global_vertex,
                               cv::cuda::PtrStepSz<Vector3f> prev_global_normal,
                               MatrixXf& A, VectorXf& b,
                               Matrix3f &curr_rotation, Vector3f &curr_translation,
                               Matrix3f &prev_rotation_inv, Vector3f &prev_translation,
                               float fX, float fY){

    float distance_threshold = 10.f;
    float angle_threshold = 20.f;

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    bool fill_zero=false;
    int img_idx = threadX + threadY * width;

    Vector3f curr_vertex = curr_frame_vertex.ptr(threadY)[threadX];
    if(isnan(curr_vertex.z()) or curr_vertex.z() < 0.f)
    {
        fill_zero=true;
        write_linear_eq(A, b, img_idx, fill_zero);
        return;
    }

    Vector3f curr_global_vertex = curr_rotation * curr_vertex + curr_translation;
    Vector2i curr_img_pixel = backproject_vertex(curr_global_vertex,
                                                 prev_rotation_inv,
                                                 prev_translation,
                                                 fX, fY);
    int curr_x = curr_img_pixel.x();
    int curr_y = curr_img_pixel.y();
    Vector3f curr_prev_global_vertex = prev_global_vertex.ptr(curr_y)[curr_x];
    Vector3f curr_prev_global_normal = prev_global_normal.ptr(curr_y)[curr_x];
    if(curr_x < 0 or curr_x >= width or curr_y < 0 or curr_y >= height
        or isnan(curr_prev_global_vertex.z())
        or isnan(curr_prev_global_normal.z()))
    {
        fill_zero=true;
        write_linear_eq(A, b, img_idx, fill_zero);
        return;
    }

    if(!isVertDistValid(curr_global_vertex, curr_prev_global_vertex, distance_threshold))
    {
        fill_zero=true;
        write_linear_eq(A, b, img_idx, fill_zero);
        return;
    }
    Vector3f curr_global_normal = curr_rotation * curr_frame_normal.ptr(threadY)[threadX];
    if(isnan(curr_global_normal.z()) or
       !isNormAngleValid(curr_global_normal, curr_prev_global_normal, angle_threshold))
    {
        fill_zero=true;
        write_linear_eq(A, b, img_idx, fill_zero);
        return;
    }

    fill_zero = false;
    write_linear_eq(A, b, img_idx, fill_zero,
                    curr_global_vertex,
                    curr_prev_global_vertex,
                    curr_prev_global_normal);

}

void point_to_plane_new( cv::cuda::GpuMat& curr_frame_vertex,
                         cv::cuda::GpuMat& curr_frame_normal,
                         cv::cuda::GpuMat& prev_global_vertex,
                         cv::cuda::GpuMat& prev_global_normal,
                         int width, int height,
                         MatrixXf A, VectorXf b,
                         Matrix3f curr_rotation, Vector3f curr_translation,
                         Matrix3f prev_rotation_inv, Vector3f prev_translation,
                         float fX, float fY,
                         Isometry3f& T){

    int rows = curr_frame_vertex.rows;
    int cols = curr_frame_vertex.cols;

    dim3 block(8, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    form_linear_eq_new<<<grid, block>>>(cols, rows,
                                        curr_frame_vertex,
                                        curr_frame_normal,
                                        prev_global_vertex,
                                        prev_global_normal,
                                        A, b,
                                        curr_rotation, curr_translation,
                                        prev_rotation_inv, prev_translation,
                                        fX, fY);

    assert(cudaSuccess == cudaDeviceSynchronize());
    Matrix<float,6,1> x = A.ldlt().solve(b);
    T = Isometry3f::Identity();
    T.linear() = ( AngleAxisf(x(0), Vector3f::UnitX())
                   * AngleAxisf(x(1), Vector3f::UnitY())
                   * AngleAxisf(x(2), Vector3f::UnitZ())
    ).toRotationMatrix();
    T.translation() = x.block(3,0,3,1);
}

void pose_estimate_helper_new(cv::cuda::GpuMat& curr_frame_vertex,
                              cv::cuda::GpuMat& curr_frame_normal,
                              cv::cuda::GpuMat& prev_global_vertex,
                              cv::cuda::GpuMat& prev_global_normal,
                              int width, int height,
                              MatrixXf A, VectorXf b,
                              Matrix3f& curr_rotation, Vector3f& curr_translation,
                              Matrix3f prev_rotation_inv, Vector3f prev_translation,
                              float fX, float fY){

    Isometry3f T;

    point_to_plane_new(curr_frame_vertex,
                       curr_frame_normal,
                       prev_global_vertex,
                       prev_global_normal,
                       width, height,
                       A, b,
                       curr_rotation, curr_translation,
                       prev_rotation_inv, prev_translation,
                       fX, fY,
                       T);
    // Return the new pose
    curr_rotation = T.rotation() * curr_rotation;
    curr_translation = T.translation() + T.rotation() * curr_translation;

}

void pose_estimate_new(const std::vector<int>&  iterations,
                       SurfaceLevelData* surf_data,
                       Pose* pose_struct)
{
    int level = surf_data->level - 1;

    Pose prev_pose = {
            pose_struct->m_trajectory,
            pose_struct->m_trajectoryInv,
    };

    Matrix3f current_rotation = pose_struct->m_trajectory.block<3, 3>(0, 0);
    Vector3f current_translation = pose_struct->m_trajectory.block<3, 1>(0, 3);

    Matrix3f prev_rotation = pose_struct->m_trajectory.block<3, 3>(0, 0);
    Matrix3f prev_rotation_inv = prev_rotation.inverse();
    Vector3f prev_translation = pose_struct->m_trajectory.block<3, 1>(0, 3);

    for ( int i = level; i >= 0; i--) {
        int iteration = iterations[i];
        unsigned int width = surf_data->level_img_width[i];
        unsigned int height = surf_data->level_img_height[i];
        float fX = surf_data->level_fX[i];
        float fY = surf_data->level_fY[i];
        for ( int j = 0; j < iteration; j++) {
            MatrixXf A(width*height, 6);
            VectorXf b(width*height);
            A.setZero();
            b.setZero();

//            cv::cuda::GpuMat curr_global_vertex_map;
//            cv::cuda::GpuMat curr_camera_vertex_map;
//            cv::cuda::GpuMat curr_global_normal_map;
//            cv::cuda::createContinuous(width, height, CV_32FC3, curr_global_vertex_map);
//            cv::cuda::createContinuous(width, height, CV_32FC3, curr_camera_vertex_map);
//            cv::cuda::createContinuous(width, height, CV_32FC3, curr_global_normal_map);
//            curr_global_vertex_map.setTo(0);
//            curr_camera_vertex_map.setTo(0);
//            curr_global_normal_map.setTo(0);

            cv::cuda::GpuMat& curr_frame_vertex = surf_data->vertex_map[i];  // in frame coordinate system
            cv::cuda::GpuMat& curr_frame_normal = surf_data->normal_map[i];  // in frame coordinate system
            cv::cuda::GpuMat& prev_global_vertex = surf_data->vertex_map_predicted[i];  // in global coordinate system, from previous frame
            cv::cuda::GpuMat& prev_global_normal = surf_data->normal_map_predicted[i];  // in global coordinate system, from previous frame

            pose_estimate_helper_new(curr_frame_vertex,
                                     curr_frame_normal,
                                     prev_global_vertex,
                                     prev_global_normal,
                                     width, height,
                                     A, b,
                                     current_rotation, current_translation,
                                     prev_rotation_inv, prev_translation,
                                     fX, fY);
        }

    }

    pose_struct->m_trajectory.block<3, 3>(0, 0) = current_rotation;
    pose_struct->m_trajectory.block<3, 1>(0, 3) = current_translation;
}

void pose_estimate(const std::vector<int>&  iterations,
                   ImageConstants* imageConstants,
                   ImageData* imageData, SurfaceLevelData* surf_data,
                   Pose* pose_struct){

    int level = surf_data->level - 1;

    std::vector<cv::cuda::GpuMat> global_vertex_maps;
    std::vector<cv::cuda::GpuMat> camera_vertex_maps;
    std::vector<cv::cuda::GpuMat> global_normal_maps;
    Pose prev_pose = {
            pose_struct->m_trajectory,
            pose_struct->m_trajectoryInv,
    };
    for ( int i = level; i >= 0; i--) {
        int j = 2 - i;
        unsigned int width = surf_data->level_img_width[i];
        unsigned int height = surf_data->level_img_height[i];

        global_vertex_maps.push_back(cv::cuda::createContinuous(width, height, CV_32FC3));
        camera_vertex_maps.push_back(cv::cuda::createContinuous(width, height, CV_32FC3));
        global_normal_maps.push_back(cv::cuda::createContinuous(width, height, CV_32FC3));



        cv::cuda::GpuMat& vertex_map = surf_data->vertex_map[i];  // in frame coordinate system
        cv::cuda::GpuMat& normal_map = surf_data->normal_map[i];  // in frame coordinate system
        cv::cuda::GpuMat& vertex_map_predicted = surf_data->vertex_map_predicted[i];  // in global coordinate system, from previous frame
        cv::cuda::GpuMat& normal_map_predicted = surf_data->normal_map_predicted[i];  // in global coordinate system, from previous frame
        cv::cuda::GpuMat& global_vertex_map = global_vertex_maps[j];  // will be filled in global coordinate sytem in following functions
        cv::cuda::GpuMat& camera_vertex_map = camera_vertex_maps[j];  // will be filled in global coordinate sytem in following functions
        cv::cuda::GpuMat& global_normal_map = global_normal_maps[j];  // will be filled in global coordinate sytem in following functions

        //dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

        Matrix3f rotation = pose_struct->m_trajectory.block<3, 3>(0, 0);
        Vector3f translation = pose_struct->m_trajectory.block<3, 1>(0, 3);

        init_global_map(imageData->m_depthMap, vertex_map, normal_map, global_vertex_map,
                        camera_vertex_map, global_normal_map,
                        rotation, translation, width, height, level);

        pose_estimate_helper(iterations[i], imageData->m_depthMap, vertex_map, vertex_map_predicted,
                             normal_map_predicted, global_vertex_map, global_normal_map, width, height,
                             rotation, translation);
        pose_struct->m_trajectory.block<3, 3>(0, 0) = rotation;
        pose_struct->m_trajectory.block<3, 1>(0, 3) = translation;
    }
}

void pose_estimate_helper( int iteration,
                           cv::cuda::GpuMat depth_map,
                           cv::cuda::GpuMat vertex_map,
                           cv::cuda::GpuMat vertex_map_predicted,
                           cv::cuda::GpuMat normal_map_predicted,
                           cv::cuda::GpuMat global_vertex_map,
                           cv::cuda::GpuMat global_normal_map,
                           int width, int height,
                           Matrix3f &rotation, Vector3f &translation){

    Isometry3f T;

    for ( int j = 0; j < iteration; j++) {
        point_to_plane( global_vertex_map, vertex_map_predicted, normal_map_predicted,
                        global_normal_map,depth_map,
                        width, height, T);
        // Return the new pose
        rotation = T.rotation() * rotation;
        translation = T.translation() + T.rotation() * translation;
    }
}


//https://www.cs.princeton.edu/~smr/papers/icpstability.pdf
//https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf

void point_to_plane( cv::cuda::GpuMat source,
                     cv::cuda::GpuMat dest,
                     cv::cuda::GpuMat normal,
                     cv::cuda::GpuMat global_normal_map,
                     cv::cuda::GpuMat depth_map,
                     int width, int height,
                     Isometry3f& T){

    bool validity_check = false;
    Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A {};
    Eigen::Matrix<float, 6, 1> b {};

    float rows = height;
    float cols = width;


    dim3 block(8, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    check_correspondence_validity<<<grid, block>>>(width, height, dest, normal,
                                                   source, global_normal_map, depth_map, validity_check);

    if (validity_check){
        form_linear_eq<<<grid, block>>>(width, height, source, normal, dest, A, b);
    }
    assert(cudaSuccess == cudaDeviceSynchronize());
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

    if ( currDepthValue > 0.f && !isnan(currDepthValue) && distance <= distance_threshold && angle <= angle_threshold ){
        validity = true;
    }
    validity = false;
}

__global__ void form_linear_eq(int width, int height,
                               cv::cuda::PtrStepSz<Vector3f> source,
                               cv::cuda::PtrStepSz<Vector3f> normal,
                               cv::cuda::PtrStepSz<Vector3f> dest,
                               Eigen::Matrix<float, 6, 6, Eigen::RowMajor> &A,
                               Eigen::Matrix<float, 6, 1> &b){
    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;
    if(source.ptr(threadY)[threadX].x() > 0.f || normal.ptr(threadY)[threadX].x() > 0.f)
        return;
    Vector3f pointToNormal = source.ptr(threadY)[threadX].cross(normal.ptr(threadY)[threadX]);
    A.block<3,3>(0,0) += pointToNormal * pointToNormal.transpose();
    A.block<3,3>(0,3) += normal.ptr(threadY)[threadX] * pointToNormal.transpose();
    A.block<3,3>(3,3) += normal.ptr(threadY)[threadX] * normal.ptr(threadY)[threadX].transpose();
    A.block<3,3>(3,0) = A.block<3,3>(0,3);

    float sum = (source.ptr(threadY)[threadX] - dest.ptr(threadY)[threadX]).dot(normal.ptr(threadY)[threadX]);
    b.head(3) -= pointToNormal * sum;
    b.tail(3) -= normal.ptr(threadY)[threadX] * sum;
}


//Get current global vertex map from surface measurement
__global__ void get_global_vertex_map( cv::cuda::PtrStepSz<float> depth_map,
                                       cv::cuda::PtrStep<Vector3f> vertex_map,
                                       cv::cuda::PtrStep<Vector3f> normal_map,
                                       Matrix3f rotation, Vector3f translation,
                                       int width, int height,
                                       cv::cuda::PtrStep<Vector3f> global_vertex_map,
                                       cv::cuda::PtrStep<Vector3f> camera_vertex_map) {

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    if(!isnan(normal_map.ptr(threadY)[threadX].x()))
    {
        global_vertex_map.ptr(threadY)[threadX] = rotation * vertex_map.ptr(threadY)[threadX] + translation;
        //camera_vertex_map.ptr(threadY)[threadX] =
    }

}

//Get current global normal map from surface measurement
__global__ void get_global_normal_map( cv::cuda::PtrStepSz<float> depth_map,
                                       cv::cuda::PtrStep<Vector3f> normal_map,
                                       Matrix3f rotation, Vector3f translation,
                                       int width, int height,
                                       cv::cuda::PtrStep<Vector3f> global_normal_map){

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    if(!isnan(normal_map.ptr(threadY)[threadX].x()))
        global_normal_map.ptr(threadY)[threadX] = rotation * normal_map.ptr(threadY)[threadX];
}

void init_global_map(cv::cuda::GpuMat depth_map,
                     cv::cuda::GpuMat vertex_map,
                     cv::cuda::GpuMat normal_map,
                     cv::cuda::GpuMat global_vertex_map,
                     cv::cuda::GpuMat camera_vertex_map,
                     cv::cuda::GpuMat global_normal_map,
                     Matrix3f rotation, Vector3f translation,
                     int width, int height, int level) {

    dim3 block(8, 8);
    float rows = height;
    float cols = width;

    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);


    get_global_vertex_map<<<grid, block>>>( depth_map, vertex_map, normal_map, rotation,
            translation, width, height, global_vertex_map, camera_vertex_map);

    get_global_normal_map<<<grid, block>>>( depth_map, normal_map, rotation,
                                            translation, width, height, global_normal_map);
    assert(cudaSuccess == cudaDeviceSynchronize());
}
