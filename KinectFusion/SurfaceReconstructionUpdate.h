#pragma once

#ifndef KINECTFUSION_SURFACE_RECONSTRUCTION_H
#define KINECTFUSION_SURFACE_RECONSTRUCTION_H

/*#include <array>
#include <algorithm>

#include <opencv4/opencv2/opencv.hpp>
//#include <opencv2/opencv.hpp>
#include "ceres/ceres.h"*/
#include <common.h>



class SurfaceReconstructionUpdate{

public:
    SurfaceReconstructionUpdate() {}


    Vector4f convert_homogeneous_vector(Vector3f vector_3d)
    {
        return Vector4f(vector_3d.x(), vector_3d.y(), vector_3d.z(), (float) 1.0);
    }


    // TODO: calculate truncation function
    float calculateSDF_truncation(float truncation_distance, float sdf){
        if (sdf >= -truncation_distance) {
            return fmin(1.f, sdf / truncation_distance); // determine threshold, 1.f currently
        }
        else return -1.f; // return - of threshold
    }

    // TODO: calculate current TSDF (FRk←calculatecurrenttsdf(Ψ,Rk,K,tg,k,p))
    //λ = ||K^-1*x||2

    //float calculateCurrentTSDF( cv::Mat depthMap, Matrix3f intrinsics, Vector3f camera_ref, int k, Vector3f p){
    float calculateCurrentTSDF(float depth, Matrix3f intrinsics, Vector3f camera_coord, Vector3f p, float truncation_distance){
        //Vector3f camera_pos = camera_ref - p;
        //float current_tsdf = (1.f / calculateLambda(intrinsics, p)) * camera_pos.norm() - depthMap.at<float>(k, 1);
        float current_tsdf = (-1.f) * ((1.f / calculateLambda(intrinsics, p)) * camera_coord.norm() - depth);
        return calculateSDF_truncation(truncation_distance, current_tsdf);
    }


    // TODO: calculate weighted running tsdf average
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

    void updateSurfaceReconstruction(ImageProperties*& image_properties){
        truncate_updated_weight = 128; // check the intuition!
        this->imageProperties = image_properties;

        for(int i=0; i < image_properties->global_tsdf->getDimX(); i++){
            for(int j=0; j < image_properties->global_tsdf->getDimY(); j++){
                for(int k=0; k < image_properties->global_tsdf->getDimZ(); k++){
                    Vector3f global_coord = Vector3f(image_properties->global_tsdf->min.x() +
                                                     image_properties->global_tsdf->diag.x() *
                                                     image_properties->global_tsdf->ddx * (float) i,
                                                     image_properties->global_tsdf->min.y() +
                                                     image_properties->global_tsdf->diag.y() *
                                                     image_properties->global_tsdf->ddy * (float) j,
                                                     image_properties->global_tsdf->min.z() +
                                                     image_properties->global_tsdf->diag.z() *
                                                     image_properties->global_tsdf->ddz * (float) k);

                    if(!global_coord.allFinite()) continue;
                    Vector3f camera_coord = image_properties->m_depthExtrinsics.block<3, 3>(0, 0)
                            * global_coord + image_properties->m_depthExtrinsics.block<3, 1>(0, 3);
                    if(!camera_coord.allFinite() && camera_coord.z() < 0) continue;
                    Vector2f image_coord = perspective_projection(camera_coord); // check the calculation is true!!

                    if(image_coord.x() < 0 || image_coord.x() >= image_properties->all_data[0].img_width
                        || image_coord.y() < 0 ||image_coord.y() >= image_properties->all_data[0].img_height)
                        continue;

                    float depth = image_properties->all_data[0].curr_level_data.at<float>(image_coord.x() + image_coord.y() *
                                                                                          image_properties->all_data[0].img_width, 1);
                    if(depth == MINF) continue;

                    float F_rk = calculateCurrentTSDF(depth, image_properties->m_depthIntrinsics, camera_coord,
                            global_coord, image_properties->truncation_distance);

                    int W_k = 1;

                    //float prev_F_rk = image_properties->global_tsdf->get(i, j, k).tsdf_distance_value;
                    Voxel prev_voxel = image_properties->global_tsdf->get(i, j, k);
                    //float prev_W_k = image_properties->global_tsdf->get(i, j, k).tsdf_weight;


                    float updated_tsdf = calculateWeightedTSDF(prev_voxel.tsdf_weight, prev_voxel.tsdf_distance_value, W_k, F_rk);
                    int truncated_weight = calculateTruncatedWeight(calculateWeightedAvgWeight
                            (prev_voxel.tsdf_weight, W_k), truncate_updated_weight);

                    //image_properties->global_tsdf->get(i, j, k).tsdf_distance_value = updated_tsdf;
                    image_properties->global_tsdf->set(i, j, k, prev_voxel);
                    //image_properties->global_tsdf->get(i, j, k).tsdf_weight = truncated_weight;
                    //image_properties->global_tsdf->set(i, j, k, prev_voxel);

                    Vector4uc curr_color;
                    if(F_rk <= image_properties->truncation_distance / 2 && F_rk >= -image_properties->truncation_distance / 2)
                    {
                        Vector4uc prev_color = image_properties->global_tsdf->get(i, j ,k).color;
                        curr_color = (Vector4uc) image_properties->m_colorMap;
                        curr_color = calculateWeightedColorUpdate(truncated_weight, prev_color, prev_voxel.tsdf_weight, curr_color);
                    }

                    Voxel* curr_voxel = new Voxel();
                    curr_voxel->tsdf_distance_value = updated_tsdf;
                    curr_voxel->tsdf_weight = truncated_weight;
                    curr_voxel->color = curr_color;
                    image_properties->global_tsdf->set(i, j, k, *curr_voxel);

                }
            }
        }

        //for(int i = 0; i < 3; i++){
        /*int i =0;
        compute_camera_ref_points(this->imageProperties, i);
        compute_global_points(this->imageProperties, i);
        int num_pixels = (int) this->imageProperties->all_data[i].img_width
                * (int) this->imageProperties->all_data[i].img_height;
        this->imageProperties->all_data[i].tsdf_value = new Voxel[num_pixels];
        Vector3f global_coord = Vector3f(this->imageProperties->global_points[0].position.x(),
                                         this->imageProperties->global_points[0].position.y(),
                                         this->imageProperties->global_points[0].position.z());
        float F_rk_init = calculateCurrentTSDF(this->imageProperties->all_data[i].curr_level_data,
                                          this->imageProperties->m_depthIntrinsics,
                                          this->imageProperties->m_trajectory.block<3, 1>(0, 3),
                                          0, global_coord);

        this->imageProperties->all_data[i].tsdf_value[0].tsdf_distance_value = F_rk_init;
        this->imageProperties->all_data[i].tsdf_value[0].tsdf_weight = 1;
        for (int j=1; j < num_pixels; j++){
            Vector3f global_coord_j = Vector3f(this->imageProperties->global_points[j].position.x(),
                                             this->imageProperties->global_points[j].position.y(),
                                             this->imageProperties->global_points[j].position.z());
            float F_rk = calculateCurrentTSDF(this->imageProperties->all_data[i].curr_level_data,
                                              this->imageProperties->m_depthIntrinsics,
                                              this->imageProperties->m_trajectory.block<3, 1>(0, 3),
                                                      j, global_coord_j);
            int W_k = 1;
            float updated_tsdf = calculateWeightedTSDF(this->imageProperties->all_data[i].tsdf_value[j-1].tsdf_weight,
                                                       this->imageProperties->all_data[i].tsdf_value[j-1].tsdf_distance_value,
                                                       W_k, F_rk);
            int truncWeight = calculateTruncatedWeight(calculateWeightedAvgWeight
                    (this->imageProperties->all_data[i].tsdf_value[j-1].tsdf_weight, 1), truncate_updated_weight);
            this->imageProperties->all_data[i].tsdf_value[j].tsdf_distance_value = updated_tsdf;
            this->imageProperties->all_data[i].tsdf_value[j].tsdf_weight = truncWeight;
        }
        //}

        }*/
        image_properties = this->imageProperties;
    }


    //HELPER FUNCTIONS
    float calculateLambda( Matrix3f intrinsics, Vector3f p){
        Vector2f projected = perspective_projection(p);
        Vector3f dot_p = Vector3f(projected.x(), projected.y(), 1.0f);
        return (intrinsics.inverse() * dot_p).norm();
    }

    Vector2f perspective_projection(Vector3f p)
    {
        Vector4f p_temp = Vector4f(p.x(), p.y(), p.z(), 1.0);
        Matrix4f identity = Matrix4f::Zero();
        identity.block<3, 3>(0, 0) = Matrix3f::Identity();
        Vector3f p2 = imageProperties->m_depthIntrinsics * identity.block<3, 4>(0, 0) * imageProperties->m_trajectoryInv * p_temp;
        return Vector2f(p2.x() / p2.z(), p2.y() / p2.z());
    }

private:

    float depth_margin;                 //μ
    float *m_depthMap;                  //Rk
    int truncate_updated_weight; // define the value to some value!
    ImageProperties* imageProperties;

};

#endif
