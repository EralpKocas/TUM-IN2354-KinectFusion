#include "Eigen.h"


class SurfaceMeasurement {

public:

    // initialize
    //SurfaceMeasurement() : { }

    bool init(float* depthMap, BYTE*colorMap, Matrix4f& trajectory,
            Matrix4f& trajectoryInv, Matrix3f& depthIntrinsics)
    {
        m_depthMap = depthMap;
        m_colorMap = colorMap;
        m_trajectory = trajectory;
        m_trajectoryInv = trajectoryInv;
        m_depthIntrinsics = depthIntrinsics;

        m_colorImageWidth = 640;
        m_colorImageHeight = 480;
        m_depthImageWidth = 640;
        m_depthImageHeight = 480;
        return true;
    }

    // compute bilateral filter

    // back-project filtered depth values to obtain vertex map

    // compute normal vectors

    // apply vertex validity mask

    // compute all multiscale

    // transorm all to global

private:
    float *m_depthMap;
    BYTE *m_colorMap;
    Matrix4f m_trajectory;
    Matrix4f m_trajectoryInv;
    Matrix3f m_depthIntrinsics;
    unsigned int m_colorImageWidth;
    unsigned int m_colorImageHeight;
    unsigned int m_depthImageWidth;
    unsigned int m_depthImageHeight;
};