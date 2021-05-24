//
// Created by eralpkocas on 24.05.21.
//

#ifndef TUM_IN2354_KINECTFUSION_COMMON_FUNCTIONS_H
#define TUM_IN2354_KINECTFUSION_COMMON_FUNCTIONS_H
#include <iostream>
#include <fstream>
#include <string>
#include "Eigen.h"

Vector3f get_translation()
{
    return Vector3f(0, 0 ,0);
}

Matrix3f get_rotation()
{
    return Matrix3f::setZero();
}


#endif //TUM_IN2354_KINECTFUSION_COMMON_FUNCTIONS_H
