#pragma once

#ifndef KINECTFUSION_SURFACE_RECONSTRUCTION_H
#define KINECTFUSION_SURFACE_RECONSTRUCTION_H

#include <common.h>



class SurfaceReconstructionUpdate{

public:
    SurfaceReconstructionUpdate() {}

    //HELPER FUNCTIONS
    float calculateLambda( Matrix3f intrinsics, Vector3f p){
        Vector2i projected = perspective_projection(p);
        Vector3f dot_p = Vector3f(projected.x(), projected.y(), 1.0f);
        return (intrinsics.inverse() * dot_p).norm();
    }

    Vector2i perspective_projection(Vector3f p)
    {
        Vector4f p_temp = Vector4f(p.x(), p.y(), p.z(), 1.0);
        Matrix4f identity = Matrix4f::Zero();
        identity.block<3, 3>(0, 0) = Matrix3f::Identity();
        Vector3f p2 = imageProperties->m_depthIntrinsics * identity.block<3, 4>(0, 0) * imageProperties->m_depthExtrinsics * imageProperties->m_trajectory * p_temp;
        return Vector2i((int) round(p2.x() / p2.z()), (int) round(p2.y() / p2.z()));
    }


    float calculateSDF_truncation(float truncation_distance, float sdf){
        if (sdf >= -truncation_distance) {
            return fmin(1.f, sdf / truncation_distance) * (sdf < 0 ? -1 : sdf > 0); // determine threshold, 1.f currently
        }
        else return -1.f; // return - of threshold
    }

    //λ = ||K^-1*x||2
    float calculateCurrentTSDF(float depth, Matrix3f intrinsics, Vector3f p, float truncation_distance){
        float current_tsdf = (1.f / calculateLambda(intrinsics, p)) * (get_translation(imageProperties) - p).norm() - depth;
        return calculateSDF_truncation(truncation_distance, current_tsdf);
    }


    // calculate weighted running tsdf average
    float calculateWeightedTSDF(int current_weight, float current_tsdf, int new_weight, float new_tsdf){
        float updated_tsdf = (current_weight * current_tsdf + new_weight * new_tsdf) /
                             (current_weight + new_weight);
        return updated_tsdf;
    }

    // TODO: calculate weighted running weight average
    int calculateWeightedAvgWeight(int current_weight, int new_weight){
        return current_weight + new_weight;
    }

    // TODO: truncate updated weight
    int calculateTruncatedWeight(int weighted_avg, int some_value){
        return std::min(weighted_avg, some_value);
    }

    Vector4uc calculateWeightedColorUpdate(int current_weight, Vector4uc curr_color, int new_weight, Vector4uc new_color)
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

    void updateSurfaceReconstruction(ImageProperties*& image_properties, Volume*& global_volume){

        //std::ofstream out("out.txt");
        //std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        //std::cout.rdbuf(out.rdbuf());

        truncate_updated_weight = 128; // check the intuition!
        this->imageProperties = image_properties;

        for(int i=0; i < global_volume->getDimX(); i++){
            for(int j=0; j < global_volume->getDimY(); j++){
                for(int k=0; k < global_volume->getDimZ(); k++){

                    Vector3f global_coord = global_volume->pos(i, j, k);

                    if(!global_coord.allFinite()) continue;

                    //Vector3f camera_coord = get_rotation(image_properties) * global_coord + get_translation(image_properties);
                    Vector3f camera_coord = (image_properties->m_depthExtrinsics * image_properties->m_trajectory * Vector4f(global_coord.x(),
                                                                                           global_coord.y(), global_coord.z(), 1.0f)).block<3,1>(0,0);
                    if(!camera_coord.allFinite() && camera_coord.z() <= 0) continue;

                    Vector2i image_coord = perspective_projection(global_coord); // check the calculation is true!!

                    if(image_coord.x() < 0 || image_coord.x() >= image_properties->all_data[0].img_width
                        || image_coord.y() < 0 ||image_coord.y() >= image_properties->all_data[0].img_height)
                        continue;

                    int index = image_coord.x() + image_coord.y() * image_properties->all_data[0].img_width;
                    float depth = image_properties->all_data[0].curr_level_data.at<float>((int) image_coord.y(),
                                                                                        (int) image_coord.x());

                    if(depth == MINF || depth <= 0) continue;


                    float F_rk = calculateCurrentTSDF(depth, image_properties->m_depthIntrinsics,
                            global_coord, image_properties->truncation_distance);

                    if(F_rk == -1.f) continue;

                    int W_k = 1;
                    Voxel prev_voxel;
                    /*if ( i == 0 || j == 0 || k == 0){
                        prev_voxel = global_volume->get(i, j, k);
                    }
                    else{
                        prev_voxel = global_volume->get(i, j, k-1);
                    }*/
                    prev_voxel = global_volume->get(i, j, k);

                    float updated_tsdf = calculateWeightedTSDF(prev_voxel.tsdf_weight, prev_voxel.tsdf_distance_value, W_k, F_rk);

                    int truncated_weight = calculateTruncatedWeight(calculateWeightedAvgWeight
                            (prev_voxel.tsdf_weight, W_k), truncate_updated_weight);

                    /*std::cout << "i: " << i << ", j: " << j << ", k: " << k << std::endl;
                    std::cout << "depth: " << depth << std::endl;
                    std::cout << "F_rk: " << F_rk << std::endl;
                    std::cout << "updated_tsdf: " << updated_tsdf << std::endl;
                    std::cout << "truncated_weight: " << truncated_weight << std::endl << std::endl;*/


                    Voxel curr_voxel;
                    curr_voxel.tsdf_distance_value = updated_tsdf;
                    curr_voxel.tsdf_weight = truncated_weight;

                    Vector4uc curr_color;
                    if(F_rk <= image_properties->truncation_distance / 2 && F_rk >= -image_properties->truncation_distance / 2)
                    {
                        // TODO: check here!!
                        Vector4uc prev_color = global_volume->get(i, j ,k).color;
                        curr_color = Vector4uc(image_properties->m_colorMap[index],
                                            image_properties->m_colorMap[index+1],
                                            image_properties->m_colorMap[index+2],
                                            image_properties->m_colorMap[index+3]);
                        curr_color = calculateWeightedColorUpdate(prev_voxel.tsdf_weight, prev_color, W_k, curr_color);
                        curr_voxel.color = curr_color;
                    }

                    global_volume->set(i, j, k, curr_voxel);  // check whether assign is successful
                }
            }
        }

        image_properties = this->imageProperties;
    }


private:

    float depth_margin;                 //μ
    float *m_depthMap;                  //Rk
    int truncate_updated_weight; // define the value to some value!
    ImageProperties* imageProperties;

};

#endif
