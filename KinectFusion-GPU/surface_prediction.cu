#include "data_types.h"
#include "surface_prediction.h"

//WARNING: Volume stuff might not have converted to CUDA right.

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

__device__ bool gridInVolume(GlobalVolume* global_volume, Vector3f curr_grid) {
    int dx = global_volume->getDimX();
    int dy = global_volume->getDimY();
    int dz = global_volume->getDimZ();
    return !(curr_grid.x() < 1 || curr_grid.x() >= dx - 1 ||
             curr_grid.y() < 1 || curr_grid.y() >= dy - 1 ||
             curr_grid.z() < 1 || curr_grid.z() >= dz - 1);
}

__device__ float calculate_trilinear_interpolation(GlobalVolume* global_volume, Vector3f p) {
    Vector3i p_int = Vector3i((int) p.x(), (int) p.y(), (int) p.z());

    //couldn't find a way to do this one
    float c000 = global_volume->get(p_int.x(), p_int.y(), p_int.z()).tsdf_distance_value;
    float c001 = global_volume->get(p_int.x(), p_int.y(), p_int.z() + 1).tsdf_distance_value;
    float c010 = global_volume->get(p_int.x(), p_int.y() + 1, p_int.z()).tsdf_distance_value;
    float c011 = global_volume->get(p_int.x(), p_int.y() + 1, p_int.z() + 1).tsdf_distance_value;
    float c100 = global_volume->get(p_int.x() + 1, p_int.y(), p_int.z()).tsdf_distance_value;
    float c101 = global_volume->get(p_int.x() + 1, p_int.y(), p_int.z() + 1).tsdf_distance_value;
    float c110 = global_volume->get(p_int.x() + 1, p_int.y() + 1, p_int.z()).tsdf_distance_value;
    float c111 = global_volume->get(p_int.x() + 1, p_int.y() + 1, p_int.z() + 1).tsdf_distance_value;

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

__global__ void helper_compute_normal_map(int width, int height) {
    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;
    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    Vector3f curr_vertex = vertex_map_predicted.ptr(threadY)[threadX];

    if (curr_vertex.z() == 0.f || isnan(curr_vertex.z())) {
        normal_map_predicted.ptr(threadY)[threadX] = Vector3f(0.f, 0.f, 0.f);
    } else {
        Vector3f neigh_1 = Vector3f(vertex_map_predicted.ptr(threadY - 1)[threadX].x() -
                                    vertex_map_predicted.ptr(threadY + 1)[threadX].x(),
                                    vertex_map_predicted.ptr(threadY - 1)[threadX].y() -
                                    vertex_map_predicted.ptr(threadY + 1)[threadX].y(),
                                    vertex_map_predicted.ptr(threadY - 1)[threadX].z() -
                                    vertex_map_predicted.ptr(threadY + 1)[threadX].z());

        Vector3f neigh_2 = Vector3f(vertex_map_predicted.ptr(threadY)[threadX - 1].x() -
                                    vertex_map_predicted.ptr(threadY)[threadX + 1].x(),
                                    vertex_map_predicted.ptr(threadY)[threadX - 1].y() -
                                    vertex_map_predicted.ptr(threadY)[threadX + 1].y(),
                                    vertex_map_predicted.ptr(threadY)[threadX - 1].z() -
                                    vertex_map_predicted.ptr(threadY)[threadX + 1].z());

        Vector3f cross_prod = neigh_1.cross(neigh_2);
        cross_prod.normalize();
        if (cross_prod.z() > 0) cross_prod *= -1;
        normal_map_predicted.ptr(threadY)[threadX] = cross_prod;
    }
}

__global__ void predict_surface(GlobalVolume global_volume, Pose pose,
                                cv::cuda::PtrStep<Vector3f> vertex_map,
                                cv::cuda::PtrStep<Vector3f> normal_map,
                                cv::cuda::PtrStep<Vector4uc> color_map,
                                float fX, float fY, float cX, float cY,
                                int width, int height, int level) {

    //simple ray skipping (speedup):
    //near F(p)=0, the fused volume holds a good approx to true sdf from p to the nearest surf interface.
    //so using known trunc dist, march along the ray in staps size < mu while F(p) vals have +ve trunc vals

    //higher quality intersections by ray/trilin cell intersection (simple approx):
    //find param t* at which the intersect more precise: t*=t-(deltat Ft+)/(F(t+deltat)+ - Ft+)

    //predicted vertex and normal maps are computed at the interpolated location in the global frame.

    int threadX = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadX >= width or threadX < 0)
        return;

    int threadY = threadIdx.y + blockDim.y * blockIdx.y;
    if (threadY >= height or threadY < 0)
        return;

    float step_size = global_volume.truncation_distance;

    //per pixel raycast. march start from the min depth, stop with zero crossing or back face
    // +0.5 for reaching pixel centers
    Vector3f pixel_ray = calculate_pixel_raycast(rotation, translation, image_constants);
    float camera_x = ((float) (threadX + 0.5) - surf_data.level_cX) / surf_data.level_fX;  // image to camera
    float camera_y = ((float) (threadY + 0.5) - surf_data.level_cY) / surf_data.level_fY;  // image to camera
    Vector3f pixel = Vector3f(camera_x, camera_y, 1.f);

    //for point on or close to surf. interface Fk(p)=0, gradient(Fk(p))=orthogonal to zero level set.
    //so the surface normal of pixel u along which p was found deriv of SDF: v[grad(F(p))].
    //Question: Why do we calculate two of them?
    Vector3f ray_dir_2 = (rotation * pixel).normalized();
    Vector3f ray_dir = (pixel_ray - translation).normalized();

    //then scale the deriv in each dimension.
    //min and max rendering range [0.4,8] => bounded time per pixel computation for any size or complexity
    //of scene with a fixed vol resolution
    //abimiz burda get_min get_max time fonksiyonlari kullanmis bizde yok bu
    //float t = calculate_search_length(translation, pixel_ray, ray_dir);  // t
    float max_ray_length = Vector3i(global_volume->getDimX(),
                                    global_volume->getDimY(),
                                    global_volume->getDimZ()).norm();

    //Vector3f pixel_grid = global_volume->compute_grid(pixel_ray);
    //Vector3f ray_dir_grid = global_volume->compute_grid(ray_dir);

    Vector3f init_pos = Vector3f(0.f, 0.f, 0.f);
    for (float step = 0; step < max_ray_length; step += step_size * 0.5) {
        Vector3f curr_pos = translation + (float) step * ray_dir;
        Vector3f curr_grid = global_volume->compute_grid(curr_pos);

        if (!gridInVolume(global_volume, curr_grid)) continue;
        init_pos = curr_pos;
        break;
    }

    if(!init_pos.allFinite() || init_pos.x() == 0.f ||
    init_pos.y() == 0.f || init_pos.z() == 0.f) continue;

    Vector3f eye_grid = global_volume->compute_grid(init_pos);
    float tsdf = global_volume->
            get((int) eye_grid.x(), (int) eye_grid.y(), (int) eye_grid.z()).tsdf_distance_value;

    float prev_tsdf = tsdf;
    Vector3f prev_grid = eye_grid; // TODO: check this, something is seem bad!

    for (float step = 0; step < max_ray_length; step += step_size * 0.5) {
        //Vector3f curr_grid = eye_grid + (float) step * ray_dir_grid;
        Vector3f curr_pos = init_pos + (float) step * ray_dir;
        Vector3f curr_grid = global_volume->compute_grid(curr_pos);

        if (!gridInVolume(image_properties, global_volume, curr_grid)) continue;

        float curr_tsdf = global_volume->
                get((int) curr_grid.x(), (int) curr_grid.y(), (int) curr_grid.z()).tsdf_distance_value;

        if (prev_tsdf < 0.f && curr_tsdf > 0.f) break;  // zero-crossing from behind

        if (prev_tsdf > 0.f && curr_tsdf < 0.f)  // zero-crossing is found
            {
            float prev_tri_interpolated_sdf = calculate_trilinear_interpolation(global_volume,
                                                                                prev_grid);
            float curr_tri_interpolated_sdf = calculate_trilinear_interpolation(global_volume,
                                                                                curr_grid);

            Voxel before = global_volume->get(curr_grid.x(), curr_grid.y(), curr_grid.z());
            global_volume->set(translation.x(), translation.y(), translation.z(),
                               global_volume->set_occupied(curr_grid));
            Voxel after = global_volume->get(curr_grid.x(), curr_grid.y(), curr_grid.z());

            //float t_star = step - ((step_size * 0.5f * prev_tsdf)/ (curr_tsdf - prev_tsdf));
            // t_star = t - ((step_size * prev_tsdf) / (curr_tsdf - prev_tsdf))

            float t_star = step - ((step_size * 0.5f * prev_tri_interpolated_sdf)
                    / (curr_tri_interpolated_sdf - prev_tri_interpolated_sdf));

            Vector3f grid_location = translation + t_star * ray_dir;

            if (!gridInVolume(global_volume, grid_location)) break;

            Vector3f vertex = translation + t_star * ray_dir;

            vertex_map_predicted.ptr(threadY)[threadX] = vertex;
            }
        prev_tsdf = curr_tsdf;
        prev_grid = curr_grid;
    }
    helper_compute_normal_map(width, height);

}

void surface_prediction(SurfaceLevelData* surf_data, GlobalVolume global_volume, Pose pose){
    //for (int i = 0; i < surf_data->level; i++) {
        //did not change the name convention. Commented out the for loop.
        //changed this from (8,8)
        dim3 block(32, 32);

        float cols = surf_data->level_img_width[i];
        float rows = surf_data->level_img_height[i];

        //Define zero level set for models Fk=tsdf val=0
        //Question: we never set these to 0?
        cv::cuda::GpuMat& vertex_map = surf_data->vertex_map_predicted[i];
        cv::cuda::GpuMat& normal_map = surf_data->normal_map_predicted[i];
        cv::cuda::GpuMat& color_map = surf_data->color_map[i];

        float fX = surf_data->level_fX[i];
        float fY = surf_data->level_fY[i];
        float cX = surf_data->level_cX[i];
        float cY = surf_data->level_cY[i];

        dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        predict_surface<<<grid, block>>>(global_volume, pose,
                                         vertex_map,
                                         normal_map,
                                         color_map,
                                         fX, fY, cX, cY,
                                         cols, rows, i);

        cudaThreadSynchronize();
    //}
}