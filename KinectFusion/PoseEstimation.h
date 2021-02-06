#ifndef KINECTFUSION_POSEESTIMATION_H
#define KINECTFUSION_POSEESTIMATION_H

#include "common.h"

class PoseEstimation{
public:

    void estimate_pose( std::vector<int> iterations, ImageProperties* imageProperties){
        Isometry3f T;
        int level = imageProperties->num_levels;

        //TODO: Put level into ImageProperties
        for ( int i = iterations[level]; i > 0; i--){
            T = point_to_plane(get_global_vertex_map(imageProperties, level),
                               imageProperties->all_data[level].vertex_map_predicted,
                               imageProperties->all_data[level].normal_map_predicted, imageProperties, level);
            // Return the new pose
            imageProperties->m_trajectory.block(0, 0, 3, 3) = T.rotation();
            imageProperties->m_trajectory.block(0, 3, 3, 1) = T.translation();
            level--;
        }

    }


    Isometry3f point_to_plane( std::vector<Vector3f> source, std::vector<Vector3f> dest, std::vector<Vector3f> normal, ImageProperties* imageProperties, int level){

        Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A {};
        Eigen::Matrix<float, 6, 1> b {};

        for ( int i = 0; i < source.size(); i++){
            if (check_correspondence_validity(imageProperties, level, i)){
                Vector3f pointToNormal = source[i].cross(normal[i]);
                A.block<3,3>(0,0) += pointToNormal * pointToNormal.transpose();
                A.block<3,3>(0,3) += normal[i] * pointToNormal.transpose();
                A.block<3,3>(3,3) += normal[i] * normal[i].transpose();
                A.block<3,3>(3,0) = A.block<3,3>(0,3);

                float sum = (source[i] - dest[i]).dot(normal[i]);
                b.head(3) -= pointToNormal * sum;
                b.tail(3) -= normal[i] * sum;
            }
        }

        Matrix<float,6,1> x = A.ldlt().solve(b);
        Isometry3f T = Isometry3f::Identity();
        T.linear() = ( AngleAxisf(x(0), Vector3f::UnitX())
                         * AngleAxisf(x(1), Vector3f::UnitY())
                         * AngleAxisf(x(2), Vector3f::UnitZ())
        ).toRotationMatrix();
        T.translation() = x.block(3,0,3,1);
        return T;
    }


    //Check correspondences on the distance of vertices and difference in normal values
    bool check_correspondence_validity(ImageProperties* image_properties, int level, int point){
        if ( !global_vertex_map){
            *global_vertex_map = get_global_vertex_map(image_properties, level);
        }
        if ( !global_normal_map){
            *global_normal_map = get_global_normal_map(image_properties, level);
        }

        float currDepthValue =  image_properties->all_data[level].curr_smoothed_data.at<float>(point);

        //TODO: Call get_global_vertex_map and get_global_normal_map once !!!
        float distance = (image_properties->all_data[level].vertex_map_predicted[point-1] -
                (*global_vertex_map)[point]).norm();

        //const float sine = normal_current_global.cross(normal_previous_global).norm();
        float angle = (*global_normal_map)[point].
                cross(image_properties->all_data[level].normal_map_predicted[point-1]).norm();

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
    std::vector<Vector3f> *global_vertex_map = nullptr;
    std::vector<Vector3f> *global_normal_map = nullptr;

    // The distance threshold in mm
    float distance_threshold { 10.f };
    // The angle threshold in degrees
    float angle_threshold { 20.f };

};




#endif //KINECTFUSION_POSEESTIMATION_H
