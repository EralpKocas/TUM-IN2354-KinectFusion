//
// Created by Beyza Tugce Bilgic on 06.07.21.
//

#include "pose_estimation.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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

__device__ void write_linear_eq(float* A, float* b, int i, bool fill_zero,
                                const Vector3f& curr_global_vertex={0, 0, 0},
                                const Vector3f& prev_global_vertex={0, 0, 0},
                                const Vector3f& prev_global_normal={0, 0, 0})
{
    if(fill_zero)
    {
        A[i+0]=0.f;
        A[i+1]=0.f;
        A[i+2]=0.f;
        A[i+3]=0.f;
        A[i+4]=0.f;
        A[i+5]=0.f;
        b[i]=0.f;
    }
    else
    {
        A[i+0]=2.f;
        A[i+1]=1.f;
        A[i+2]=1.f;
        A[i+3]=1.f;
        A[i+4]=1.f;
        A[i+5]=1.f;
        b[i]=1.f;
    }
}

__global__ void form_linear_eq_new(int width, int height,
                               cv::cuda::PtrStepSz<Vector3f> curr_frame_vertex,
                               cv::cuda::PtrStepSz<Vector3f> curr_frame_normal,
                               cv::cuda::PtrStepSz<Vector3f> prev_global_vertex,
                               cv::cuda::PtrStepSz<Vector3f> prev_global_normal,
                               Matrix3f curr_rotation, Vector3f curr_translation,
                               Matrix3f prev_rotation_inv, Vector3f prev_translation,
                               float fX, float fY,
                               float* d_A, float* d_b){

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    float distance_threshold = 10.f;
    float angle_threshold = 20.f;
    bool fill_zero=false;
    int img_idx = threadX + threadY * width;

    Vector3f curr_vertex = curr_frame_vertex.ptr(threadY)[threadX];
    if(isnan(curr_vertex.z()) or curr_vertex.z() < 0.f)
    {
        fill_zero=true;
        write_linear_eq(d_A, d_b, img_idx, fill_zero);
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
        write_linear_eq(d_A, d_b, img_idx, fill_zero);
        return;
    }

    if(!isVertDistValid(curr_global_vertex, curr_prev_global_vertex, distance_threshold))
    {
        fill_zero=true;
        write_linear_eq(d_A, d_b, img_idx, fill_zero);
        return;
    }
    Vector3f curr_global_normal = curr_rotation * curr_frame_normal.ptr(threadY)[threadX];
    if(isnan(curr_global_normal.z()) or
       !isNormAngleValid(curr_global_normal, curr_prev_global_normal, angle_threshold))
    {
        fill_zero=true;
        write_linear_eq(d_A, d_b, img_idx, fill_zero);
        return;
    }

    fill_zero = false;
    write_linear_eq(d_A, d_b, img_idx, fill_zero,
                    curr_global_vertex,
                    curr_prev_global_vertex,
                    curr_prev_global_normal);

}

void point_to_plane_new( cv::cuda::GpuMat& curr_frame_vertex,
                         cv::cuda::GpuMat& curr_frame_normal,
                         cv::cuda::GpuMat& prev_global_vertex,
                         cv::cuda::GpuMat& prev_global_normal,
                         int width, int height,
                         Matrix3f curr_rotation, Vector3f curr_translation,
                         Matrix3f prev_rotation_inv, Vector3f prev_translation,
                         float fX, float fY,
                         Isometry3f& T){
    MatrixXf A(width*height, 6);
    VectorXf b(width*height);
    A.setZero();
    b.setZero();
    float* A_data = A.data();
    float* b_data = b.data();
//    std::cout << "before size" << std::endl;

    size_t A_bytes = (width*height*6) * sizeof(float);
    size_t b_bytes = (width*height) * sizeof(float);
//    std::cout << "before malloc" << std::endl;
    float *d_A, *d_b;
    cudaMalloc(&d_A, A_bytes);
    cudaMalloc(&d_b, b_bytes);
//    std::cout << "before memcpy" << std::endl;

    cudaMemcpy(d_A, A_data, A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data, b_bytes, cudaMemcpyHostToDevice);
//    std::cout << "before block grid" << std::endl;

    dim3 block(8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
//    std::cout << "before form linear eq" << std::endl;
    form_linear_eq_new<<<grid, block>>>(width, height,
                                        curr_frame_vertex,
                                        curr_frame_normal,
                                        prev_global_vertex,
                                        prev_global_normal,
                                        curr_rotation, curr_translation,
                                        prev_rotation_inv, prev_translation,
                                        fX, fY, d_A, d_b);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    //assert(cudaSuccess == cudaDeviceSynchronize());
//    std::cout << "before host copy" << std::endl;
//    std::cout << *A_data << std::endl;
//    std::cout << A.block<1, 1>(0, 0) << std::endl;
    cudaMemcpy(A_data, d_A, A_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_data, d_b, b_bytes, cudaMemcpyDeviceToHost);
//    std::cout << "before solving" << std::endl;
//    std::cout << *A_data << std::endl;
//    std::cout << A.block<1, 1>(0, 0) << std::endl;
//    Matrix<float,6,1> x = A.ldlt().solve(b);
    Eigen::Matrix<float, 6, 1> x { A.fullPivLu().solve(b).cast<float>() };
//    std::cout << "after solving" << std::endl;
    T = Isometry3f::Identity();
    T.linear() = ( AngleAxisf(x(0), Vector3f::UnitX())
                   * AngleAxisf(x(1), Vector3f::UnitY())
                   * AngleAxisf(x(2), Vector3f::UnitZ())).toRotationMatrix();
    T.translation() = x.block(3,0,3,1);
}

void pose_estimate_helper_new(cv::cuda::GpuMat& curr_frame_vertex,
                              cv::cuda::GpuMat& curr_frame_normal,
                              cv::cuda::GpuMat& prev_global_vertex,
                              cv::cuda::GpuMat& prev_global_normal,
                              int width, int height,
                              Matrix3f& curr_rotation, Vector3f& curr_translation,
                              Matrix3f prev_rotation_inv, Vector3f prev_translation,
                              float fX, float fY){

    Isometry3f T;
//    int rows = curr_frame_vertex.rows;
//    int cols = curr_frame_vertex.cols;
//    std::cout << "before point to plane" << std::endl;
    point_to_plane_new(curr_frame_vertex,
                       curr_frame_normal,
                       prev_global_vertex,
                       prev_global_normal,
                       curr_frame_vertex.cols, curr_frame_vertex.rows,
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
//            std::cout << "before pose helper" << std::endl;
            pose_estimate_helper_new(curr_frame_vertex,
                                     curr_frame_normal,
                                     prev_global_vertex,
                                     prev_global_normal,
                                     width, height,
                                     current_rotation, current_translation,
                                     prev_rotation_inv, prev_translation,
                                     fX, fY);

        }

    }
    pose_struct->m_trajectory.block<3, 3>(0, 0) = current_rotation;
    pose_struct->m_trajectory.block<3, 1>(0, 3) = current_translation;
    pose_struct->m_trajectoryInv = pose_struct->m_trajectory.inverse();
}