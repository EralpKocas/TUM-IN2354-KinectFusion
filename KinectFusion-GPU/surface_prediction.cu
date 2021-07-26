#include "data_types.h"
#include "surface_prediction.h"

//WARNING: Volume stuff might not have converted to CUDA right.
#define DIVSHORTMAX 0.0000305185f //1.f / SHRT_MAX;

__device__ Vector3f calculate_pixel_ray_cast(Matrix3f rotation, Vector3f translation, ImageConstants imageConstants) {
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    //img to cam
    float camera_x = (threadX - imageConstants.cX) / imageConstants.fX;
    float camera_y = (threadY - imageConstants.cY) / imageConstants.fY;
    Vector3f camera_vec = Vector3f(camera_x, camera_y, 1.f);
    //cam to global
    Vector3f ray_cast = rotation * camera_vec + translation;

    return ray_cast;
}

// P = O + t * R where R is normalized ray direction and O is eye, translation vector in our case.
// Then, t = (P - O) / R
__device__ float calculate_search_length(Vector3f eye, Vector3f ray_dir) {
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    int threadZ = blockIdx.z * blockDim.z + threadIdx.z;
    float t_x = (threadX - eye.x()) / ray_dir.x();
    float t_y = (threadY - eye.y()) / ray_dir.y();
    float t_z = (threadZ - eye.z()) / ray_dir.z();

    return fmax(fmax(fabs(t_x), fabs(t_y)), fabs(t_z));
}

__device__ bool gridInVolume(int volume_size, Vector3f curr_grid) {
    int dx = volume_size;
    int dy = volume_size;
    int dz = volume_size;
    return (curr_grid.x() < 1 || curr_grid.x() >= dx - 1 ||
             curr_grid.y() < 1 || curr_grid.y() >= dy - 1 ||
             curr_grid.z() < 1 || curr_grid.z() >= dz - 1);
}

__device__ float calculate_trilinear_interpolation(cv::cuda::PtrStepSz<float> tsdf_values,
                                                   int volume_size,Vector3f p) {
    Vector3i p_int = Vector3i((int) p.x(), (int) p.y(), (int) p.z());

    float c000 = tsdf_values.ptr(p_int.z()*volume_size+p_int.y())[p_int.x()];
    float c001 = tsdf_values.ptr((p_int.z()+1)*volume_size+p_int.y())[p_int.x()];
    float c010 = tsdf_values.ptr(p_int.z()*volume_size+p_int.y()+1)[p_int.x()];
    float c011 = tsdf_values.ptr((p_int.z()+1)*volume_size+p_int.y()+1)[p_int.x()];
    float c100 = tsdf_values.ptr(p_int.z()*volume_size+p_int.y())[p_int.x() +1];
    float c101 = tsdf_values.ptr((p_int.z()+1)*volume_size+p_int.y())[p_int.x() +1];
    float c110 = tsdf_values.ptr(p_int.z()*volume_size+p_int.y()+1)[p_int.x() +1];
    float c111 = tsdf_values.ptr((p_int.z()+1)*volume_size+p_int.y()+1)[p_int.x() +1];

    float xd = p.x() - p_int.x();
    float yd = p.y() - p_int.y();
    float zd = p.z() - p_int.z();

    float c00 = c000 * (1 - xd) + c100 * xd;
    float c01 = c001 * (1 - xd) + c101 * xd;
    float c10 = c010 * (1 - xd) + c110 * xd;
    float c11 = c011 * (1 - xd) + c111 * xd;

    float c0 = c00 * (1 - yd) + c10 * yd;
    float c1 = c01 * (1 - yd) + c11 * yd;

    float c = c0 * (1 - zd) + c1 * zd;

    return c;
}

__device__ Vector3f compute_grid(Vector3f p, Vector3f min, Vector3f max,int volume_size)
{
    return Vector3f(((p[0] - min.x()) / (max.x() - min.x())) / volume_size,
                    ((p[1] - min.y()) / (max.y() - min.y())) / volume_size,
                    ((p[2] - min.z()) / (max.z() - min.z())) / volume_size
    );
}

__device__ Vector3f compute_grid_new(Vector3f p, Vector3f min, Vector3f max,int volume_size)
{
    return Vector3f(((p[0] - min.x()) / (max.x() - min.x())) / volume_size,
                    ((p[1] - min.y()) / (max.y() - min.y())) / volume_size,
                    ((p[2] - min.z()) / (max.z() - min.z())) / volume_size
    );
}

__device__ Vector3f compute_normal_vector(cv::cuda::PtrStep<Vector3f> vertex_map,
                                          Vector3f curr_grid, int volume_size)
{

    int base_idxFirst = (int)(curr_grid.z()*volume_size +curr_grid.y());
    int base_idxSecond = (int)curr_grid.x();

    Vector3f curr_vertex = vertex_map.ptr(base_idxFirst)[base_idxSecond];

    if (curr_vertex.z() == 0.f) {
        return Vector3f(0.f, 0.f, 0.f);
        //TODO: add color as 0, 0 ,0 ,0 if necessary??
    } else {
        Vector3f neigh_1 = Vector3f(vertex_map.ptr(base_idxFirst)[base_idxSecond].x() -
                                    vertex_map.ptr(base_idxFirst + 1)[base_idxSecond].x(),
                                    vertex_map.ptr(base_idxFirst - 1)[base_idxSecond].y() -
                                    vertex_map.ptr(base_idxFirst + 1)[base_idxSecond].y(),
                                    vertex_map.ptr(base_idxFirst - 1)[base_idxSecond].z() -
                                    vertex_map.ptr(base_idxFirst + 1)[base_idxSecond].z());

        Vector3f neigh_2 = Vector3f(vertex_map.ptr(base_idxFirst)[base_idxSecond - 1].x() -
                                    vertex_map.ptr(base_idxFirst)[base_idxSecond + 1].x(),
                                    vertex_map.ptr(base_idxFirst)[base_idxSecond - 1].y() -
                                    vertex_map.ptr(base_idxFirst)[base_idxSecond + 1].y(),
                                    vertex_map.ptr(base_idxFirst)[base_idxSecond - 1].z() -
                                    vertex_map.ptr(base_idxFirst)[base_idxSecond + 1].z());

        Vector3f cross_prod = neigh_1.cross(neigh_2);
        cross_prod.normalize();
        if (cross_prod.z() > 0) cross_prod *= -1;
        return cross_prod;
        //TODO: add color
    }
}

__device__ Vector3f compute_normal_vector_new(Vector3f vertex, cv::cuda::PtrStepSz<float> tsdf_values, float step_size,
                                              int volume_size)
{
    Vector3i vertex_x_h = Vector3i((int) (vertex.x() - step_size * 0.5),
                                 (int) (vertex.y()),
                                 (int) (vertex.z()));
    Vector3i vertex_x_h_ = Vector3i((int) (vertex.x() + step_size * 0.5),
                                  (int) (vertex.y()),
                                  (int) (vertex.z()));

    Vector3i vertex_y_h = Vector3i((int) (vertex.x()),
                                   (int) (vertex.y() - step_size * 0.5),
                                   (int) (vertex.z()));
    Vector3i vertex_y_h_ = Vector3i((int) (vertex.x()),
                                    (int) (vertex.y() + step_size * 0.5),
                                    (int) (vertex.z()));

    Vector3i vertex_z_h = Vector3i((int) (vertex.x()),
                                   (int) (vertex.y()),
                                   (int) (vertex.z() - step_size * 0.5));
    Vector3i vertex_z_h_ = Vector3i((int) (vertex.x()),
                                    (int) (vertex.y()),
                                    (int) (vertex.z() + step_size * 0.5));
    Vector3f tsdf_h = Vector3f(
            tsdf_values.ptr((int)(vertex_x_h.z()*volume_size +vertex_x_h.y()))[(int)vertex_x_h.x()],
            tsdf_values.ptr((int)(vertex_y_h.z()*volume_size +vertex_y_h.y()))[(int)vertex_y_h.x()],
            tsdf_values.ptr((int)(vertex_z_h.z()*volume_size +vertex_z_h.y()))[(int)vertex_z_h.x()]
            );
    Vector3f tsdf_h_ = Vector3f(
            tsdf_values.ptr((int)(vertex_x_h_.z()*volume_size +vertex_x_h_.y()))[(int)vertex_x_h_.x()],
            tsdf_values.ptr((int)(vertex_y_h_.z()*volume_size +vertex_y_h_.y()))[(int)vertex_y_h_.x()],
            tsdf_values.ptr((int)(vertex_z_h_.z()*volume_size +vertex_z_h_.y()))[(int)vertex_z_h_.x()]
    );
    return tsdf_h - tsdf_h_;

}
__device__ __forceinline__
float get_min_time(const float3& volume_max, const Vector3f& origin, const Vector3f& direction)
{
    float txmin = ((direction.x() > 0 ? 0.f : volume_max.x) - origin.x()) / direction.x();
    float tymin = ((direction.y() > 0 ? 0.f : volume_max.y) - origin.y()) / direction.y();
    float tzmin = ((direction.z() > 0 ? 0.f : volume_max.z) - origin.z()) / direction.z();

    return fmax(fmax(txmin, tymin), tzmin);
}

__device__ __forceinline__
float get_max_time(const float3& volume_max, const Vector3f& origin, const Vector3f& direction)
{
    float txmax = ((direction.x() > 0 ? volume_max.x : 0.f) - origin.x()) / direction.x();
    float tymax = ((direction.y() > 0 ? volume_max.y : 0.f) - origin.y()) / direction.y();
    float tzmax = ((direction.z() > 0 ? volume_max.z : 0.f) - origin.z()) / direction.z();

    return fmin(fmin(txmax, tymax), tzmax);
}

__global__ void predict_surface(
                                cv::cuda::PtrStepSz<float> tsdf_values,
                                cv::cuda::PtrStepSz<float> tsdf_weights,
                                cv::cuda::PtrStepSz<Vector4uc> tsdf_color,
                                cv::cuda::PtrStep<Vector3f> vertex_map,
                                cv::cuda::PtrStep<Vector3f> normal_map,
                                cv::cuda::PtrStep<Vector4uc> color_map,
                                float fX, float fY, float cX, float cY,
                                int width, int height, int level,
                                float truncation_distance,Matrix4f pose_traj,
                                Vector3f min, Vector3f max,int volume_size,
                                float voxel_scale) {

    //predicted vertex and normal maps are computed at the interpolated location in the global frame.
    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height)
        return;

    float step_size = truncation_distance;
    //per pixel raycast. march start from the min depth, stop with zero crossing or back face
    // +0.5 for reaching pixel centers
    //Vector3f pixel_ray = calculate_pixel_raycast(rotation, translation, image_constants);
    float camera_x = (((float) threadX) - cX) / fX;  // image to camera
    float camera_y = (((float) threadY) - cY) / fY;  // image to camera

    Matrix3f rotation = pose_traj.block<3, 3>(0, 0);
    Vector3f translation = pose_traj.block<3, 1>(0, 3);
    Vector3f pixel_ray = rotation * Vector3f(camera_x, camera_y, 1.f) + translation;
    //for point on or close to surf. interface Fk(p)=0, gradient(Fk(p))=orthogonal to zero level set.
    //so the surface normal of pixel u along which p was found deriv of SDF: v[grad(F(p))].
    //Question: Why do we calculate two of them?
//    Vector3f ray_dir = (rotation * pixel_ray).normalized();
    Vector3f ray_dir = (pixel_ray - translation);
    ray_dir.normalize();

    //then scale the deriv in each dimension.
    //min and max rendering range [0.4,8] => bounded time per pixel computation for any size or complexity
    //of scene with a fixed vol resolution
    //simple ray skipping (speedup):
    //near F(p)=0, the fused volume holds a good approx to true sdf from p to the nearest surf interface.
    //so using known trunc dist, march along the ray in steps size < mu while F(p) vals have +ve trunc vals
//    printf("printing translation pos\n");
//    printf("translation x: %f\n", translation.x());
//    printf("translation y: %f\n", translation.y());
//    printf("translation z: %f\n", translation.z());
//    printf("step size: %f\n", step_size);
//    printf("===============================\n");
//    printf("printing curr pos\n");
//    printf("curr pos x: %f\n", curr_pos.x());
//    printf("curr pos y: %f\n", curr_pos.y());
//    printf("curr pos z: %f\n", curr_pos.z());
//    printf("===============================\n");
//    Vector3f curr_grid = curr_pos / voxel_scale;

//    if (gridInVolume( volume_size, curr_grid))
//        prev_tsdf = tsdf_values.ptr((int)(curr_grid.z()*volume_size +curr_grid.y()))[(int)curr_grid.x()];
//    else
//        return;
//    Vector3f init_pos = pixel_ray;
//    Vector3f prev_grid = pixel_ray / voxel_scale;
    float len_x = (((float) volume_size * voxel_scale - translation.x()) / ray_dir.x());
    float len_y = (((float) volume_size * voxel_scale - translation.y()) / ray_dir.y());
    float len_z = (((float) volume_size * voxel_scale - translation.z()) / ray_dir.z());
    if(len_x < 0) len_x = -1 * len_x;
    if(len_y < 0) len_y = -1 * len_y;
    if(len_z < 0) len_z = -1 * len_z;
    float ray_length;
    if(len_x < len_y){
        ray_length = len_x;
    }
    else
        ray_length = len_y;
    if(len_z < ray_length)
        ray_length = len_z;
    Vector3f curr_pos = translation + ray_length * ray_dir;
    Vector3f curr_grid = (translation + (ray_dir * ray_length)) / voxel_scale;
    float curr_tsdf = tsdf_values.ptr((int)(curr_grid.z()*volume_size +curr_grid.y()))[(int)curr_grid.x()];
    const float max_ray_length = ray_length + (float) volume_size * sqrt(2.f);
    for (int step = 0; step < max_ray_length; step += step_size * 0.5) {
        Vector3f prev_grid = curr_grid;
        Vector3f curr_pos = translation + (float) (step) * ray_dir;
        Vector3f curr_grid = curr_pos / voxel_scale;
        printf("1111\n");
        if (gridInVolume( volume_size, curr_grid)) continue;
        printf("2222\n");
        const float prev_tsdf = curr_tsdf;

        float curr_tsdf = tsdf_values.ptr((int)(curr_grid.z()*volume_size +curr_grid.y()))[(int)curr_grid.x()];

        printf("1111\n");
        if (prev_tsdf < 0.f && curr_tsdf > 0.f) break;  // zero-crossing from behind
        printf("2222\n");

        if (prev_tsdf > 0.f && curr_tsdf < 0.f)  // zero-crossing is found
        {
            printf("3333\n");
            //higher quality intersections by ray/trilin cell intersection (simple approx):
            float prev_tri_interpolated_sdf = calculate_trilinear_interpolation(tsdf_values, volume_size,
                                                                                prev_grid);
            float curr_tri_interpolated_sdf = calculate_trilinear_interpolation(tsdf_values, volume_size,
                                                                                curr_grid);
            printf("4444\n");

            float _tsdf = tsdf_values.ptr((int)(curr_grid.z()*volume_size +curr_grid.y()))[(int)curr_grid.x()];
            float _weight = tsdf_weights.ptr((int)(curr_grid.z()*volume_size +curr_grid.y()))[(int)curr_grid.x()];
            Vector4uc _color = tsdf_color.ptr((int)(curr_grid.z()*volume_size +curr_grid.y()))[(int)curr_grid.x()];
            printf("5555\n");

            tsdf_values.ptr((int)(translation.z()*volume_size +translation.y()))[(int)translation.x()] = _tsdf;
            tsdf_weights.ptr((int)(translation.z()*volume_size +translation.y()))[(int)translation.x()] = _weight;
            tsdf_color.ptr((int)(translation.z()*volume_size +translation.y()))[(int)translation.x()] = _color;
            printf("6666\n");

            // TODO:: Commented out because we are not using this anywhere
//            Voxel after = global_volume->get(curr_grid.x(), curr_grid.y(), curr_grid.z());

            //float t_star = step - ((step_size * 0.5f * prev_tsdf)/ (curr_tsdf - prev_tsdf));
            // t_star = t - ((step_size * prev_tsdf) / (curr_tsdf - prev_tsdf))

            //find param t* at which the intersect more precise: t*=t-(deltat Ft+)/(F(t+deltat)+ - Ft+)
            float t_star = step - ((step_size * 0.5f * prev_tri_interpolated_sdf)
                    / (curr_tri_interpolated_sdf - prev_tri_interpolated_sdf));
            printf("7777\n");

            //no idea if the vector subtraction works like this but,
            //do the normal calculation
//            Vector3f normal = curr_tri_interpolated_sdf-prev_tri_interpolated_sdf;
//            if (normal.norm() == 0) break;

//            if (normal.norm() == 0) break;
//            normal.normalize();

            Vector3f vertex = translation + t_star * ray_dir;
            printf("8888\n");

//            if (!gridInVolume(volume_size, vertex / voxel_scale)) break;
//            Vector3f normal = compute_normal_vector(vertex_map,curr_grid,volume_size);
            Vector3f normal = compute_normal_vector_new(vertex, tsdf_values, step_size,
                    volume_size);
            printf("9999\n");

            vertex_map.ptr(threadY)[threadX] = vertex;
            normal_map.ptr(threadY)[threadX] = normal;
            color_map.ptr(threadY)[threadX] =
                    tsdf_color.ptr((int)(curr_grid.z()*volume_size +curr_grid.y()))[(int)curr_grid.x()];
            printf("1010\n");

        }
    }
}

void surface_prediction(SurfaceLevelData* surf_data, GlobalVolume* global_volume, Pose pose){
    for (int i = 0; i < surf_data->level; i++) {
        dim3 block(8, 8);

        cv::cuda::GpuMat& tsdf_vals = global_volume->TSDF_values;
        cv::cuda::GpuMat& tsdf_weights = global_volume->TSDF_weight;
        cv::cuda::GpuMat& tsdf_color = global_volume->TSDF_color;

        cv::cuda::GpuMat& vertex_map = surf_data->vertex_map_predicted[i];
        cv::cuda::GpuMat& normal_map = surf_data->normal_map_predicted[i];
        cv::cuda::GpuMat& color_map = surf_data->color_map_predicted[i];

        vertex_map.setTo(0);
        normal_map.setTo(0);
        color_map.setTo(0);
        int cols = surf_data->vertex_map_predicted[i].cols;
        int rows = surf_data->vertex_map_predicted[i].rows;

        float fX = surf_data->level_fX[i];
        float fY = surf_data->level_fY[i];
        float cX = surf_data->level_cX[i];
        float cY = surf_data->level_cY[i];
        std::cout << "heyo!" << std::endl;
        dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        predict_surface<<<grid, block>>>(tsdf_vals,tsdf_weights,tsdf_color,
                                         vertex_map,
                                         normal_map,
                                         color_map,
                                         fX, fY, cX, cY,
                                         cols, rows, i,
                                         global_volume->truncation_distance, pose.m_trajectory,
                                         global_volume->min, global_volume->max,global_volume->volume_size.x,
                                         global_volume->voxel_scale);
        std::cout << "bye!" << std::endl;
        assert(cudaSuccess == cudaDeviceSynchronize());
    }
}