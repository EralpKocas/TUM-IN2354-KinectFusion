#include <array>

#include "Eigen.h"


class SurfaceMeasurement {

public:

    // initialize
    //SurfaceMeasurement() : { }
    struct CameraRefPoints
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // position stored as 4 floats (4th component is supposed to be 1.0)
        Vector3f position;
        // color stored as 4 unsigned char
        Vector4uc color;
    };

    struct GlobalPoints
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // position stored as 4 floats (4th component is supposed to be 1.0)
        Vector4f position;
        // color stored as 4 unsigned char
        Vector4uc color;
    };

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

        camera_reference_points = new CameraRefPoints[m_depthImageWidth * m_depthImageHeight];
        global_points = new GlobalPoints[m_depthImageWidth * m_depthImageHeight];
        return true;
    }
    // compute 3d camera reference points
    void compute_camera_ref_points()
    {
        int numWH = m_depthImageWidth * m_depthImageHeight;

        float fX = m_depthIntrinsics(0, 0);
        float fY = m_depthIntrinsics(1, 1);
        float cX = m_depthIntrinsics(0, 2);
        float cY = m_depthIntrinsics(1, 2);

        for(int i=0; i < numWH; i++){
            if(m_depthMap[i] == MINF){
                camera_reference_points[i].position = Vector3f(MINF, MINF, MINF);
                global_points[i].color = Vector4uc(0,0,0,0);
            }
            else{
                int pixel_y = i / m_depthImageWidth;
                int pixel_x = i - pixel_y * m_depthImageWidth;
                float currDepthValue = m_depthMap[i];
                float camera_x = currDepthValue * ((float) pixel_x - cX) / fX;
                float camera_y = currDepthValue * ((float) pixel_y - cY) / fY;

                camera_reference_points[i].position = Vector3f(camera_x, camera_y, currDepthValue);
                camera_reference_points[i].color = Vector4uc(m_colorMap[4*i], m_colorMap[4*i+1], m_colorMap[4*i+2], m_colorMap[4*i+3]);
            }
        }
        for(int i=0; i < numWH; i++) {
            if (m_depthMap[i] != MINF) {
                std::cout << "first camera ref point at i: \n" << i << std::endl;
                std::cout << "curr depth: \n" << m_depthMap[i] << std::endl;
                std::cout << "camera_reference_points: \n" << camera_reference_points[i].position << std::endl;
                break;
            }
        }
    }

    // compute global 3D points
    void compute_global_points()
    {
        int numWH = m_depthImageWidth * m_depthImageHeight;

        for(int i=0; i < numWH; i++) {
            if (m_depthMap[i] == MINF) {
                global_points[i].position = Vector4f(MINF, MINF, MINF, MINF);
                global_points[i].color = Vector4uc(0, 0, 0, 0);
            } else {
                Vector4f camera_ref_vector = Vector4f(camera_reference_points[i].position.x(),
                                                      camera_reference_points[i].position.y(),
                                                      camera_reference_points[i].position.z(),
                                                      (float) 1.0);

                Vector4f global_point = m_trajectory * camera_ref_vector;

                global_points[i].position = global_point;
                global_points[i].color = camera_reference_points[i].color;
            }
        }
        for(int i=0; i < numWH; i++) {
            if (m_depthMap[i] != MINF) {
                std::cout << "first global point at i: \n" << i << std::endl;
                std::cout << "m_trajectory: \n" << m_trajectory << std::endl;
                std::cout << "global: \n" << global_points[i].position << std::endl;
                break;
            }
        }
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
    CameraRefPoints *camera_reference_points;
    GlobalPoints *global_points;
};