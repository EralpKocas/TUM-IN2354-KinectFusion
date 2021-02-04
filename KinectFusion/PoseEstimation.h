#ifndef KINECTFUSION_POSEESTIMATION_H
#define KINECTFUSION_POSEESTIMATION_H

#include "common.h"

class PoseEstimation{
public:

    void estimate_step(){

    }

    //TODO: Iterative solution for trajectory by minimising the energy form around the previous estimate of trajectory
    void camera_pose_estimate(){

    }

    //Check correspondences on the distance of vertices and difference in normal values
    bool check_correspondence_validity(ImageProperties* image_properties, int level, int point){
        float currDepthValue =  image_properties->all_data[level].curr_smoothed_data.at<float>(point);

        //TODO: Call get_global_vertex_map and get_global_normal_map once !!!
        float distance = (image_properties->all_data[level].vertex_map_predicted[point] -
                get_global_vertex_map(image_properties, level)[point]).norm();

        //const float sine = normal_current_global.cross(normal_previous_global).norm();
        float angle = get_global_normal_map(image_properties, level)[point].
                cross(image_properties->all_data[level].normal_map_predicted[point]).norm();

        if ( currDepthValue != MINF && distance <= distance_threshold && angle <= angle_threshold ){
            return true;
        }
        return false;
    }

    //Get current global vertex map from surface measurement
    std::vector<Vector3f> get_global_vertex_map(ImageProperties* image_properties, int level){
        Matrix3f rotation = image_properties->m_trajectory.block<3, 3>(0, 0);
        Vector3f translation = image_properties->m_trajectory.block<3, 1>(0, 3);

        std::vector<Vector3f> global_vertex_map;
        int curr_width = (int) image_properties->all_data[level].img_width;
        int curr_height = (int) image_properties->all_data[level].img_height;
        int numWH = curr_width * curr_height;

        for (int i=0; i < numWH; i++){
            float currDepthValue =  image_properties->all_data[level].curr_smoothed_data.at<float>(i);
            if(currDepthValue == MINF){
                global_vertex_map[i] = Vector3f(MINF, MINF, MINF);
            }
            else{
                global_vertex_map[i] = rotation * image_properties->all_data[level].vertex_map[i] + translation;

            }
        }
        return global_vertex_map;
    }

    //Get current global normal map from surface measurement
    std::vector<Vector3f> get_global_normal_map(ImageProperties* image_properties, int level){
        Matrix3f rotation = image_properties->m_trajectory.block<3, 3>(0, 0);
        Vector3f translation = image_properties->m_trajectory.block<3, 1>(0, 3);

        std::vector<Vector3f> global_normal_map;
        int curr_width = (int) image_properties->all_data[level].img_width;
        int curr_height = (int) image_properties->all_data[level].img_height;
        int numWH = curr_width * curr_height;

        for (int i=0; i < numWH; i++){
            float currDepthValue =  image_properties->all_data[level].curr_smoothed_data.at<float>(i);
            if(currDepthValue == MINF){
                global_normal_map[i] = Vector3f(MINF, MINF, MINF);
            }
            else{
                global_normal_map[i] = rotation * image_properties->all_data[level].normal_map[i];
            }
        }
        return global_normal_map;
    }




private:
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A {};
    Eigen::Matrix<double, 6, 1> b {};

    // The distance threshold in mm
    float distance_threshold { 10.f };
    // The angle threshold in degrees
    float angle_threshold { 20.f };

};




#endif //KINECTFUSION_POSEESTIMATION_H
