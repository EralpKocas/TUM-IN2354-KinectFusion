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
            float new_tsdf = fmin(1.f, sdf / truncation_distance);
            return new_tsdf;
        }
        return 0;
    }

    // TODO: calculate current TSDF (FRk←calculatecurrenttsdf(Ψ,Rk,K,tg,k,p))
    //λ = ||K^-1*x||2

    float calculateCurrentTSDF( cv::Mat depthMap, Matrix3f intrinsics, Vector3f camera_ref, int k, Vector3f p){
        Vector3f camera_pos = camera_ref - p;
        float current_tsdf = (1.f / calculateLambda(intrinsics, p)) * camera_pos.norm() - depthMap.at<float>(k, 1);
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

    void updateSurfaceReconstruction(ImageProperties*& image_properties){
        truncation_distance = 1.0; // find the correct value!
        truncate_updated_weight = 128; // check the intuition!
        this->imageProperties = image_properties;
        for(int i = 0; i < 3; i++){
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
        }

        int non_zero = 0;
        for(int k=0; k < image_properties->all_data[0].img_width * image_properties->all_data[0].img_height; k++){
            if(!isnan(image_properties->all_data[0].vertex_map[k].x())){
                non_zero = k;
                break;
            }
        }
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
    float truncation_distance;          //η
    float *m_depthMap;                  //Rk
    int truncate_updated_weight; // define the value to some value!
    ImageProperties* imageProperties;

};

#endif
