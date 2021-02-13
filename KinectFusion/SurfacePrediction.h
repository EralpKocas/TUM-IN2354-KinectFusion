#pragma once

#ifndef KINECTFUSION_SURFACE_PREDICTION_H
#define KINECTFUSION_SURFACE_PREDICTION_H

#include <array>

#include <common.h>


class SurfacePrediction {

public:
    SurfacePrediction() {}

    // a function to calculate raycast of a pixel
    Vector3f calculate_pixel_raycast(Vector3f pixel, Matrix3f rotation, Vector3f translation,
            float fX, float fY, float cX, float cY)
    {
        float camera_x = ((float) pixel.x() - cX) / fX;  // image to camera
        float camera_y = ((float) pixel.y() - cY) / fY;  // image to camera
        return rotation * Vector3f(camera_x, camera_y, 1.f) + translation;  // camera to global
    }

    Vector3f calculate_raycast_dir(Vector3f eye, Vector3f current_ray)
    {
        return (eye - current_ray).normalized();
    }

    // P = O + t * R where R is normalized ray direction and O is eye, translation vector in our case.
    // Then, t = (P - O) / R
    float calculate_search_length(Vector3f eye, Vector3f pixel, Vector3f ray_dir)
    {
        float t_x = (pixel.x() - eye.x()) / ray_dir.x();
        float t_y = (pixel.y() - eye.y()) / ray_dir.y();
        float t_z = (pixel.z() - eye.z()) / ray_dir.z();

        return fmax(fmax(fabs(t_x), fabs(t_y)), fabs(t_z));
    }

    bool gridInVolume(ImageProperties*& image_properties, Vector3f curr_grid)
    {
        int dx = image_properties->global_tsdf->getDimX();
        int dy = image_properties->global_tsdf->getDimY();
        int dz = image_properties->global_tsdf->getDimZ();

        return !(curr_grid.x() < 1 || curr_grid.x() >= dx - 1 ||
                 curr_grid.y() < 1 || curr_grid.y() >= dy - 1 ||
                 curr_grid.z() < 1 || curr_grid.z() >= dz - 1);
    }

    float calculate_trilinear_interpolation(ImageProperties*& image_properties, Vector3f p)
    {
        Vector3i p_int = Vector3i((int) p.x(), (int) p.y(), (int) p.z());

        float c000 = image_properties->global_tsdf->get(p_int.x(), p_int.y(), p_int.z()).tsdf_distance_value;
        float c001 = image_properties->global_tsdf->get(p_int.x(), p_int.y(), p_int.z() + 1).tsdf_distance_value;
        float c010 = image_properties->global_tsdf->get(p_int.x(), p_int.y() + 1, p_int.z()).tsdf_distance_value;
        float c011 = image_properties->global_tsdf->get(p_int.x(), p_int.y() + 1, p_int.z() + 1).tsdf_distance_value;
        float c100 = image_properties->global_tsdf->get(p_int.x() + 1, p_int.y(), p_int.z()).tsdf_distance_value;
        float c101 = image_properties->global_tsdf->get(p_int.x() + 1, p_int.y(), p_int.z() + 1).tsdf_distance_value;
        float c110 = image_properties->global_tsdf->get(p_int.x() + 1, p_int.y() + 1, p_int.z()).tsdf_distance_value;
        float c111 = image_properties->global_tsdf->get(p_int.x() + 1, p_int.y() + 1, p_int.z() + 1).tsdf_distance_value;

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

    void helper_compute_vertex_map(ImageProperties*& image_properties, int level)
    {
        int curr_width = (int) image_properties->all_data[level].img_width;
        int curr_height = (int) image_properties->all_data[level].img_height;
        int numWH = curr_width * curr_height;

        for(int i=0; i < numWH; i++) {
            Vector3f curr_vertex = image_properties->all_data[level].vertex_map[i];
            if (curr_vertex.z() == MINF || isnan(curr_vertex.z())) {
                image_properties->all_data[level].vertex_map_predicted[i] = Vector3f(MINF, MINF, MINF);
            } else {
                int pixel_y = i / curr_width;
                int pixel_x = i - pixel_y * curr_width;

                int right_pixel = (pixel_x + 1) + pixel_y * curr_width;
                int left_pixel = (pixel_x - 1) + pixel_y * curr_width;
                int bottom_pixel = pixel_x + (pixel_y + 1) * curr_width;
                int upper_pixel = pixel_x + (pixel_y - 1) * curr_width;

                Vector3f neigh_1 = Vector3f(image_properties->all_data[level].vertex_map_predicted[left_pixel].x() -
                                            image_properties->all_data[level].vertex_map_predicted[right_pixel].x(),
                                            image_properties->all_data[level].vertex_map_predicted[left_pixel].y() -
                                            image_properties->all_data[level].vertex_map_predicted[right_pixel].y(),
                                            image_properties->all_data[level].vertex_map_predicted[left_pixel].z() -
                                            image_properties->all_data[level].vertex_map_predicted[right_pixel].z());

                Vector3f neigh_2 = Vector3f(image_properties->all_data[level].vertex_map_predicted[upper_pixel].x() -
                                            image_properties->all_data[level].vertex_map_predicted[bottom_pixel].x(),
                                            image_properties->all_data[level].vertex_map_predicted[upper_pixel].y() -
                                            image_properties->all_data[level].vertex_map_predicted[bottom_pixel].y(),
                                            image_properties->all_data[level].vertex_map_predicted[upper_pixel].z() -
                                            image_properties->all_data[level].vertex_map_predicted[bottom_pixel].z());

                Vector3f cross_prod = neigh_1.cross(neigh_2);
                cross_prod.normalize();
                if(cross_prod.z() > 0) cross_prod *= -1;

                image_properties->all_data[level].normal_map_predicted[i] = cross_prod;
            }
        }
    }

    void predict_surface(ImageProperties*& image_properties)
    {
        for(int level=0; level < image_properties->num_levels; level++)
        {
            Vector3f translation = get_translation(image_properties);
            Matrix3f rotation = get_rotation(image_properties);

            float step_size = image_properties->truncation_distance;
            int width = (int) image_properties->all_data[level].img_width;
            int height = (int) image_properties->all_data[level].img_height;
            for(int i=0; i < width; i++)
            {
                for(int j=0; j < height; j++)
                {
                    // +0.5 for reaching pixel centers
                    Vector3f pixel_ray = calculate_pixel_raycast(Vector3f(float(i+0.5), float(j+0.5), 1.f), rotation, translation,
                            image_properties->all_data[level].curr_fX, image_properties->all_data[level].curr_fY,
                            image_properties->all_data[level].curr_cX, image_properties->all_data[level].curr_cY);

                    Vector3f ray_dir = calculate_raycast_dir(translation, pixel_ray);

                    //float t = calculate_search_length(translation, pixel_ray, ray_dir);  // t
                    float max_ray_length = Vector3i(image_properties->global_tsdf->getDimX(),
                                                    image_properties->global_tsdf->getDimY(),
                                                    image_properties->global_tsdf->getDimZ()).norm();

                    //Vector3f pixel_grid = image_properties->global_tsdf->compute_grid(pixel_ray);
                    //Vector3f ray_dir_grid = image_properties->global_tsdf->compute_grid(ray_dir);
                    Vector3f eye_grid = image_properties->global_tsdf->compute_grid(translation);

                    float tsdf = image_properties->global_tsdf->
                            get((int) eye_grid.x(), (int) eye_grid.y(), (int) eye_grid.z()).tsdf_distance_value;

                    float prev_tsdf = tsdf;
                    Vector3f prev_grid = eye_grid; // TODO: check this, something is seem bad!

                    for(float step=0; step < max_ray_length; step+=step_size*0.5)
                    {
                        //Vector3f curr_grid = eye_grid + (float) step * ray_dir_grid;
                        Vector3f curr_grid = translation + (float) step * ray_dir;

                        if(!gridInVolume(image_properties, curr_grid)) continue;

                        float curr_tsdf = image_properties->global_tsdf->
                                get((int) curr_grid.x(), (int) curr_grid.y(), (int) curr_grid.z()).tsdf_distance_value;

                        if(prev_tsdf < 0.f && curr_tsdf > 0.f) break;  // zero-crossing from behind

                        if(prev_tsdf > 0.f && curr_tsdf < 0.f)  // zero-crossing is found
                        {
                            float prev_tri_interpolated_sdf = calculate_trilinear_interpolation(image_properties, prev_grid);
                            float curr_tri_interpolated_sdf = calculate_trilinear_interpolation(image_properties, curr_grid);

                            //TODO: CHECK HERE
                            Voxel before = image_properties->global_tsdf->get(translation.x(), translation.y(), translation.z());
                            image_properties->global_tsdf->set(translation.x(), translation.y(), translation.z(), image_properties->global_tsdf->set_occupied(translation));
                            Voxel after = image_properties->global_tsdf->get(translation.x(), translation.y(), translation.z());

                            //float t_star = step - ((step_size * 0.5f * prev_tsdf)/ (curr_tsdf - prev_tsdf));
                            // t_star = t - ((step_size * prev_tsdf) / (curr_tsdf - prev_tsdf))

                            float t_star = step - ((step_size * 0.5f * prev_tri_interpolated_sdf)
                                    / (curr_tri_interpolated_sdf - prev_tri_interpolated_sdf));

                            Vector3f grid_location = translation + t_star * ray_dir;

                            if(!gridInVolume(image_properties, grid_location)) break;

                            Vector3f vertex = translation + t_star * ray_dir;

                            image_properties->all_data[level].vertex_map_predicted[j*width + i] = vertex;
                            //image_properties->all_data[level].normal_map_predicted[j*width + i] = normal;
                        }
                        prev_tsdf = curr_tsdf;
                        prev_grid = curr_grid;
                    }
                }
            }
            helper_compute_vertex_map(image_properties, level);
        }
    }

private:

};

#endif
