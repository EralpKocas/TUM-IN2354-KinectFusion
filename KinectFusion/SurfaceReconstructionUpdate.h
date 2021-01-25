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

    float calculateCurrentTSDF( float* depthMap, Matrix3f intrinsics, Vector3f camera_ref, int k, Vector3f p){
        Vector3f camera_pos = camera_ref - p;
        float current_tsdf = (1.f / calculateLambda(intrinsics, p)) * camera_pos.norm() - depthMap[k];
        return calculateSDF_truncation(truncation_distance, current_tsdf);
    }


    // TODO: calculate weighted running tsdf average
    float calculateWeightedTSDF(int current_weight, float current_tsdf, int new_weight, float new_tsdf){
        float updated_tsdf = (current_weight * current_tsdf + new_weight * new_tsdf) /
                             (current_weight + new_weight);
    }

    // TODO: calculate weighted running weight average
    float calculateWeightedAvgWeight(int current_weight, int new_weight){
        return current_weight + new_weight;
    }

    // TODO: truncate updated weight
    int calculateTruncatedWeight(int current_weight, int new_weight, int some_value){
        return std::min(current_weight + new_weight, some_value);
    }

    void updateSurfaceReconstruction(){

    }


    //HELPER FUNCTIONS
    float calculateLambda( Matrix3f intrinsics, Vector3f p){
        Vector2f projected = perspective_projection(p);
        Vector3f dot_p = Vector3f(projected.x(), projected.y(), 1.0f);
        return (intrinsics.inverse() * dot_p).norm();
    }

    Vector2f perspective_projection(Vector3f p)
    {
        Vector3f p2 = imageProperties->m_depthIntrinsics * (imageProperties->m_depthExtrinsics).inverse() * p;
        return Vector2f(p2.x() / p2.z(), p2.y() / p2.z());
    }

private:

    float depth_margin;                 //μ
    float truncation_distance;          //η
    float *m_depthMap;                  //Rk

    ImageProperties* imageProperties;

};

#endif
