#include "data_types.h"
#include "surface_prediction.h"

__device__ Vector3f calculate_pixel_ray_cast(Vector3f pixel, Matrix3f rotation, Vector3f translation,
                                 ImageConstants *&imageConstants, Vector3f eye, Volume *&global_volume,
                                 Vector3f curr_grid) {
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    int threadZ = blockIdx.z * blockDim.z + threadIdx.z;

    //img to cam
    float camera_x = (threadX - imageConstants.cX) / imageConstants.fX;
    float camera_y = (threadY - imageConstants.cY) / imageConstants.fY;
    Vector3f camera_vec = Vector3f(camera_x, camera_y, 1.f);
    //cam to global
    Vector3f raycast = rotation * camera_vec + translation;

    return raycast;

    /*
    //calculate_raycast_dir
    Vector3f ray_dir = (raycast - eye).normalized();

    //calculate_search_length
    float t_x = (threadX - eye.x()) / ray_dir.x();
    float t_y = (threadY - eye.y()) / ray_dir.y();
    float t_z = (threadZ - eye.z()) / ray_dir.z();
    float search_length = fmax(fmax(fabs(t_x), fabs(t_y)), fabs(t_z));

    //place to gridInVolume
    if (curr_grid.x() < 1 || curr_grid.x() >= global_volume.x - 1 ||
        curr_grid.y() < 1 || curr_grid.y() >= global_volume.y - 1 ||
        curr_grid.z() < 1 || curr_grid.z() >= global_volume.z - 1)
        break;
    */
}

__device__ Vector3f calculate_raycast_dir(Vector3f eye, Vector3f current_ray) {
    return (current_ray - eye).normalized();
}

__device__ bool gridInVolume(ImageProperties *&image_properties, Volume *&global_volume, Vector3f curr_grid) {
    int dx = global_volume->getDimX();
    int dy = global_volume->getDimY();
    int dz = global_volume->getDimZ();
    return !(curr_grid.x() < 1 || curr_grid.x() >= global_volume.x - 1 ||
             curr_grid.y() < 1 || curr_grid.y() >= global_volume.y - 1 ||
             curr_grid.z() < 1 || curr_grid.z() >= global_volume.z - 1);
}


// P = O + t * R where R is normalized ray direction and O is eye, translation vector in our case.
// Then, t = (P - O) / R
__device__ float calculate_search_length(Vector3f eye, Vector3f pixel, Vector3f ray_dir) {
    float t_x = (pixel.x() - eye.x()) / ray_dir.x();
    float t_y = (pixel.y() - eye.y()) / ray_dir.y();
    float t_z = (pixel.z() - eye.z()) / ray_dir.z();

    return fmax(fmax(fabs(t_x), fabs(t_y)), fabs(t_z));
}


__device__ float calculate_trilinear_interpolation(ImageConstants *&image_constants, GlobalVolume *&global_volume, Vector3f p) {
    Vector3i p_int = Vector3i((int) p.x(), (int) p.y(), (int) p.z());

    //not sure if this is the right way to write it, had problem with the syntax lol
    float c000 = global_volume->get(p_int.x(), p_int.y(), p_int.z()).TSDF_values;
    float c001 = global_volume->get(p_int.x(), p_int.y(), p_int.z() + 1).TSDF_values;
    float c010 = global_volume->get(p_int.x(), p_int.y() + 1, p_int.z()).TSDF_values;
    float c011 = global_volume->get(p_int.x(), p_int.y() + 1, p_int.z() + 1).TSDF_values;
    float c100 = global_volume->get(p_int.x() + 1, p_int.y(), p_int.z()).TSDF_values;
    float c101 = global_volume->get(p_int.x() + 1, p_int.y(), p_int.z() + 1).TSDF_values;
    float c110 = global_volume->get(p_int.x() + 1, p_int.y() + 1, p_int.z()).TSDF_values;
    float c111 = global_volume->get(p_int.x() + 1, p_int.y() + 1, p_int.z() + 1).TSDF_values;

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

__global__ void helper_compute_normal_map(SurfaceLevelData *&surface_level_data) {
    int curr_width = surface_level_data->level_img_width
    int curr_height = surface_level_data->level_img_height
    int numWH = curr_width * curr_height;

    for (int i = 1; i < curr_height - 1; i++) {
        for (int j = 1; j < curr_width - 1; j++) {
            Vector3f curr_vertex = surface_level_data->vertex_map_predicted[j + i * curr_width];
            if (curr_vertex.z() == MINF || isnan(curr_vertex.z())) {
                surface_level_data->normal_map_predicted[j + i * curr_width] = Vector3f(MINF, MINF, MINF);
            } else {
                int pixel_y = i;
                int pixel_x = j;

                int right_pixel = (pixel_x + 1) + pixel_y * curr_width;
                int left_pixel = (pixel_x - 1) + pixel_y * curr_width;
                int bottom_pixel = pixel_x + (pixel_y + 1) * curr_width;
                int upper_pixel = pixel_x + (pixel_y - 1) * curr_width;

                Vector3f neigh_1 = Vector3f(
                        surface_level_data->vertex_map_predicted[left_pixel].x() -
                        surface_level_data->vertex_map_predicted[right_pixel].x(),
                        surface_level_data->vertex_map_predicted[left_pixel].y() -
                        surface_level_data->vertex_map_predicted[right_pixel].y(),
                        surface_level_data->vertex_map_predicted[left_pixel].z() -
                        surface_level_data->vertex_map_predicted[right_pixel].z());

                Vector3f neigh_2 = Vector3f(
                        surface_level_data->vertex_map_predicted[upper_pixel].x() -
                        surface_level_data->vertex_map_predicted[bottom_pixel].x(),
                        surface_level_data->vertex_map_predicted[upper_pixel].y() -
                        surface_level_data->vertex_map_predicted[bottom_pixel].y(),
                        surface_level_data->vertex_map_predicted[upper_pixel].z() -
                        surface_level_data->vertex_map_predicted[bottom_pixel].z());

                Vector3f cross_prod = neigh_1.cross(neigh_2);
                cross_prod.normalize();
                if (cross_prod.z() > 0) cross_prod *= -1;

                surface_level_data->normal_map_predicted[j + i * curr_width] = cross_prod;
            }
        }
    }
}

__global__ void predict_surface(ImageConstants *&image_constants, SurfaceLevelData *&surface_level_date, GlobalVolume *&global_volume) {
    for (int level = 0; level < image_constants->num_levels; level++) {
        Vector3f translation = get_translation(image_constants);
        cv::cuda::GpuMat rotation = get_rotation(image_constants);

        float step_size = image_constants->truncation_distance;
        int width = (int) surface_level_date->level_img_width;
        int height = (int) surface_level_date->level_img_height;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                // +0.5 for reaching pixel centers
                Vector3f pixel_ray = calculate_pixel_raycast(Vector3f(float(i + 0.5), float(j + 0.5), 1.f),
                                                             rotation, translation,
                                                             surface_level_data.level_fX,
                                                             surface_level_data.level_fY,
                                                             surface_level_data.level_cX,
                                                             surface_level_data.level_cY);
                float camera_x = ((float) (i + 0.5) - surface_level_data.level_cX) /
                                 surface_level_data.level_fX;  // image to camera
                float camera_y = ((float) (j + 0.5) - surface_level_data.level_cY) /
                                 surface_level_data.level_fY;  // image to camera
                Vector3f pixel = Vector3f(camera_x, camera_y, 1.f);
                Vector3f ray_dir_2 = (rotation * pixel).normalized();
                Vector3f ray_dir = calculate_raycast_dir(translation, pixel_ray);

                //float t = calculate_search_length(translation, pixel_ray, ray_dir);  // t
                float max_ray_length = Vector3i(global_volume.TSDF_values[0],
                                                global_volume.TSDF_values[1],
                                                global_volume.TSDF_values[2].norm();

                //Vector3f pixel_grid = global_volume->compute_grid(pixel_ray);
                //Vector3f ray_dir_grid = global_volume->compute_grid(ray_dir);

                //TODO: I AM HERE
                Vector3f init_pos = Vector3f(MINF, MINF, MINF);
                for (float step = 0; step < max_ray_length; step += step_size * 0.5) {
                    Vector3f curr_pos = translation + (float) step * ray_dir;
                    Vector3f curr_grid = global_volume->compute_grid(curr_pos);

                    if (!gridInVolume(image_properties, global_volume, curr_grid)) continue;
                    init_pos = curr_pos;
                    break;
                }

                if(!init_pos.allFinite() || init_pos.x() == MINF ||
                   init_pos.y() == MINF || init_pos.z() == MINF) continue;

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
                        float prev_tri_interpolated_sdf = calculate_trilinear_interpolation(image_properties,
                                                                                            global_volume,
                                                                                            prev_grid);
                        float curr_tri_interpolated_sdf = calculate_trilinear_interpolation(image_properties,
                                                                                            global_volume,
                                                                                            curr_grid);

                        //TODO: CHECK HERE
                        Voxel before = global_volume->get(curr_grid.x(), curr_grid.y(), curr_grid.z());
                        global_volume->set(translation.x(), translation.y(), translation.z(),
                                           global_volume->set_occupied(curr_grid));
                        Voxel after = global_volume->get(curr_grid.x(), curr_grid.y(), curr_grid.z());

                        //float t_star = step - ((step_size * 0.5f * prev_tsdf)/ (curr_tsdf - prev_tsdf));
                        // t_star = t - ((step_size * prev_tsdf) / (curr_tsdf - prev_tsdf))

                        float t_star = step - ((step_size * 0.5f * prev_tri_interpolated_sdf)
                                               / (curr_tri_interpolated_sdf - prev_tri_interpolated_sdf));

                        Vector3f grid_location = translation + t_star * ray_dir;

                        if (!gridInVolume(image_constants, global_volume, grid_location)) break;

                        Vector3f vertex = translation + t_star * ray_dir;

                        surface_level_data->vertex_map_predicted[j * width + i] = vertex;
                        //image_properties->all_data[level].normal_map_predicted[j*width + i] = normal;
                    }
                    prev_tsdf = curr_tsdf;
                    prev_grid = curr_grid;
                }
            }
        }
        helper_compute_normal_map(image_constants, level);
    }
}