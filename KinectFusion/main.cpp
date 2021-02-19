#include <iostream>
//#include <fstream>
#include <array>

#include <iostream>
#include <fstream>
#include <string>

#include <common.h>
#include "VirtualSensor_freiburg.h"

#include "SurfaceMeasurement.h"
#include "SurfaceReconstructionUpdate.h"
#include "PoseEstimation.h"
#include "SurfacePrediction.h"
#include "SimpleMesh.h"
#include "MarchingCubes.h"

#define MIN_POINT -1.5f, -1.0f, -0.1f
#define MAX_POINT 1.5f, 1.0f, 3.5f
//#define MIN_POINT -0.1f, -0.1f, -0.1f
//#define MAX_POINT 1.1f, 1.1f, 1.1f

#define RESOLUTION 50, 50, 50

ImageProperties* init(VirtualSensor_freiburg &sensor)
{
    ImageProperties* imageProperties = new ImageProperties();

    imageProperties->depthMap = sensor.getDepth();
    imageProperties->prev_depthMap = new float[640 * 480];

    imageProperties->m_depthMap = cv::Mat(640, 480, CV_32F, sensor.getDepth());
    imageProperties->m_colorMap = sensor.getColorRGBX();
    //imageProperties->m_colorMap = cv::Mat(640*480, 1, CV_8UC4, sensor.getColorRGBX());
    imageProperties->m_trajectory = sensor.getTrajectory();
    imageProperties->m_trajectoryInv = sensor.getTrajectory().inverse();
    imageProperties->m_depthIntrinsics = sensor.getDepthIntrinsics();
    imageProperties->m_depthExtrinsics = sensor.getDepthExtrinsics();

    imageProperties->fX = sensor.getDepthIntrinsics().coeffRef(0,0);
    imageProperties->fY = sensor.getDepthIntrinsics().coeffRef(1,1);
    imageProperties->cX = sensor.getDepthIntrinsics().coeffRef(0,2);
    imageProperties->cY = sensor.getDepthIntrinsics().coeffRef(1,2);

    imageProperties->m_colorImageWidth = 640;
    imageProperties->m_colorImageHeight = 480;
    imageProperties->m_depthImageWidth = 640;
    imageProperties->m_depthImageHeight = 480;

    imageProperties->truncation_distance = 25.f; // find the correct value!

    //imageProperties->camera_reference_points = new CameraRefPoints*[imageProperties->num_levels];

    //imageProperties->camera_reference_points = new CameraRefPoints[imageProperties->m_depthImageWidth * imageProperties->m_depthImageHeight];

    //imageProperties->global_points = new GlobalPoints*[imageProperties->num_levels];

    //imageProperties->global_points = new GlobalPoints[imageProperties->m_depthImageWidth * imageProperties->m_depthImageHeight];
    //imageProperties->global_tsdf = new Volume(Vector3f(MIN_POINT), Vector3f(MAX_POINT), RESOLUTION, 3);
    return imageProperties;
}

void initSurfaceLevelData(ImageProperties* imageProperties){
    imageProperties->all_data = new SurfaceLevelData[imageProperties->num_levels];
    for(int i = 0; i < imageProperties->num_levels; i++){
        imageProperties->all_data[i].img_width = imageProperties->m_depthImageWidth;
        imageProperties->all_data[i].img_height = imageProperties->m_depthImageHeight;
    }
}

bool isDistValid(Vector3f p1, Vector3f p2, Vector3f p3, float edgeThreshold){
    float x1 = p1.x();
    float y1 = p1.y();
    float z1 = p1.z();

    float x2 = p2.x();
    float y2 = p2.y();
    float z2 = p2.z();

    float x3 = p3.x();
    float y3 = p3.y();
    float z3 = p3.z();

    float dist1 = pow((pow((x1-x2), 2)+ pow((y1-y2), 2) + pow((z1-z2), 2)), (0.5));
    float dist2 = pow((pow((x1-x3), 2)+ pow((y1-y3), 2) + pow((z1-z3), 2)), (0.5));
    float dist3 = pow((pow((x2-x3), 2)+ pow((y2-y3), 2) + pow((z2-z3), 2)), (0.5));

    if (dist1 < edgeThreshold and dist2 < edgeThreshold and dist3 < edgeThreshold) return TRUE;
    return FALSE;
}

bool WriteMesh(ImageProperties*& image_properties, unsigned int width, unsigned int height, const std::string& filename)
{
    float edgeThreshold = 0.01f; // 1cm

    // TODO 2: use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
    // - have a look at the "off_sample.off" file to see how to store the vertices and triangles
    // - for debugging we recommend to first only write out the vertices (set the number of faces to zero)
    // - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
    // - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
    // - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
    // - only write triangles with valid vertices and an edge length smaller then edgeThreshold

    // TODO: Get number of vertices
    unsigned int nVertices = width * height;

    // TODO: Determine number of valid faces
    unsigned nFaces = 0;

    for(int i=0; i < nVertices; i++){
        image_properties->all_data[0].vertex_map[i] = image_properties->m_trajectory.block(0, 0, 3, 3)
                * image_properties->all_data[0].vertex_map[i] + image_properties->m_trajectory.block(0, 3, 3, 1);
        if (i != nVertices - 1 and image_properties->all_data[0].vertex_map[i].x() != MINF){
            if (i % width == width-1 or i >= nVertices - width) continue;

            int corner_2 = i + width;
            int corner_3 = i + 1;
            image_properties->all_data[0].vertex_map[corner_2] = image_properties->m_trajectory.block(0, 0, 3, 3)
                    * image_properties->all_data[0].vertex_map[corner_2] + image_properties->m_trajectory.block(0, 3, 3, 1);
            image_properties->all_data[0].vertex_map[corner_3] = image_properties->m_trajectory.block(0, 0, 3, 3)
                                                                  * image_properties->all_data[0].vertex_map[corner_3] +
                    image_properties->m_trajectory.block(0, 3, 3, 1);

            bool distTrue = isDistValid(image_properties->all_data[0].vertex_map[i],
                    image_properties->all_data[0].vertex_map[corner_2],
                    image_properties->all_data[0].vertex_map[corner_3], edgeThreshold);
            if (image_properties->all_data[0].vertex_map[corner_2].x() != MINF and
            image_properties->all_data[0].vertex_map[corner_3].x() != MINF and distTrue){
                nFaces++;
            } else continue;

            int corner_4 = corner_2 + 1;
            image_properties->all_data[0].vertex_map[corner_4] = image_properties->m_trajectory.block(0, 0, 3, 3)
                    * image_properties->all_data[0].vertex_map[corner_4] + image_properties->m_trajectory.block(0, 3, 3, 1);
            distTrue = isDistValid(image_properties->all_data[0].vertex_map[corner_2],
                    image_properties->all_data[0].vertex_map[corner_3],
                    image_properties->all_data[0].vertex_map[corner_4], edgeThreshold);
            if (image_properties->all_data[0].vertex_map[corner_4].x() != MINF and distTrue)
                nFaces++;
        }
    }
    // Write off file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) return false;

    // write header
    outFile << "COFF" << std::endl;
    outFile << nVertices << " " << nFaces << " 0" << std::endl;

    // TODO: save vertices

    // TODO: save valid faces

    for(int i=0; i < nVertices; i++){
        Vector4f temp_vertices = Vector4f();

        if(image_properties->all_data[0].vertex_map[i].x() == MINF){
            temp_vertices = Vector4f(0, 0, 0, 1);
            //temp_vertices->color = vertices[i].color;

            outFile << temp_vertices.x() << " " << temp_vertices.y() << " " << temp_vertices.z()
                    << " " << (int) image_properties->m_colorMap[4*i] << " " << (int)image_properties->m_colorMap[4*i+1] << " "
                    << (int)image_properties->m_colorMap[4*i+2] << " " << (int)image_properties->m_colorMap[4*i+3] << std::endl;
        }
        else{
            outFile << image_properties->all_data[0].vertex_map[i].x() << " "
            << image_properties->all_data[0].vertex_map[i].y() << " "
            << image_properties->all_data[0].vertex_map[i].z()
                    << " " << (int)image_properties->m_colorMap[4*i] << " " <<(int) image_properties->m_colorMap[4*i+1]<< " "
                    << (int)image_properties->m_colorMap[4*i+2] << " " << (int)image_properties->m_colorMap[4*i+3] << std::endl;
        }
    }

    for(int i=0; i < nVertices; i++){
        if (i != nVertices - 1 and image_properties->all_data[0].vertex_map[i].x() != MINF){
            if (i % width == width-1 or i >= nVertices - width) continue;

            int corner_2 = i + width;
            int corner_3 = i + 1;
            bool distTrue = isDistValid(image_properties->all_data[0].vertex_map[i],
                    image_properties->all_data[0].vertex_map[corner_2],
                    image_properties->all_data[0].vertex_map[corner_3], edgeThreshold);
            if (image_properties->all_data[0].vertex_map[corner_2].x() != MINF
            and image_properties->all_data[0].vertex_map[corner_3].x() != MINF and distTrue){
                outFile << "3 " << i << " " << corner_2 << " " << corner_3 << std::endl;
            } else continue;

            int corner_4 = corner_2 + 1;
            distTrue = isDistValid(image_properties->all_data[0].vertex_map[corner_2], image_properties->all_data[0].vertex_map[corner_3],
                                   image_properties->all_data[0].vertex_map[corner_4], edgeThreshold);
            if (image_properties->all_data[0].vertex_map[corner_4].x() != MINF and distTrue)
                outFile << "3 " << corner_2 << " " << corner_4 << " " << corner_3 << std::endl;
        }
    }

    // close file
    outFile.close();

    return true;
}



int main() {

    // Make sure this path points to the data folder
    //std::string filenameIn = "/Users/beyzatugcebilgic/Desktop/TUM-IN2354-KinectFusion/KinectFusion/data/rgbd_dataset_freiburg1_xyz/";
    std::string filenameIn = "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/data/rgbd_dataset_freiburg1_xyz/";
    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor_freiburg sensor;
    if (!sensor.init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    Volume *global_volume = new Volume(Vector3f(MIN_POINT), Vector3f(MAX_POINT), RESOLUTION, 3);

    // convert video to meshes
    int i = 1;
    Matrix3f prev_frame_rotation;
    Vector3f prev_frame_translation;
    ImageProperties* prev_imageProperties;
    while (sensor.processNextFrame()) {
        ImageProperties *imageProperties = init(sensor);

        //TODO:: Discuss this!
        //initSurfaceLevelData(imageProperties);

        //compute_camera_ref_points(imageProperties);
        //compute_global_points(imageProperties);

        // get ptr to the current depth frame
        // depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())

        //float *depthMap = imageProperties->m_depthMap;

        // get ptr to the current color frame
        // color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
        //BYTE *colorMap = imageProperties->m_colorMap;

        std::vector<int> iterations = {4, 5, 10};

        /*std::ofstream out("out4.txt");
        std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        std::cout.rdbuf(out.rdbuf());*/

        // check applying first homework -> done!
        SurfaceMeasurement surface_measurement;
        surface_measurement.surface_measurement_pipeline(imageProperties);

        if ( i != 1){
            // TODO : check applying point cloud estimation and comparison
            PoseEstimation pose_estimation;
            pose_estimation.estimate_pose(iterations, prev_imageProperties, imageProperties);
        }

        // test surface measurement
        std::string filenameBaseOut = "mesh_";
        std::stringstream ss;

        ss << filenameBaseOut << i << ".off";
        if (!WriteMesh(imageProperties, 640, 480, ss.str())) {
            std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
            return -1;
        }
        prev_imageProperties = imageProperties;
        //imageProperties->prev_depthMap = imageProperties->depthMap;
        //prev_frame_rotation = imageProperties->m_trajectory.block(0, 0, 3, 3);
        //prev_frame_translation = imageProperties->m_trajectory.block(0, 3, 3, 1);
        i++;
    }
    return 0;
}
/*        if ( i != 1){
            // TODO : check applying point cloud estimation and comparison
            PoseEstimation pose_estimation;
            pose_estimation.estimate_pose(iterations, imageProperties);
        }
        // TODO: check applying marching cubes algorithm directly (line 159) on meshlab
        SurfaceReconstructionUpdate reconstruction_update;
        reconstruction_update.updateSurfaceReconstruction(imageProperties, global_volume);
*/
        /*for (unsigned int x = 0; x < global_volume->getDimX(); x++)
        {
            for (unsigned int y = 0; y < global_volume->getDimY(); y++)
            {
                for (unsigned int z = 0; z < global_volume->getDimZ(); z++)
                {
                    std::cout << "at x: << x" " y: " << y  << "z: "  << z << std::endl;
                    std::cout << "tsdf: " << global_volume->get(x, y, z).tsdf_distance_value << std::endl;
                    std::cout << "weight: " << global_volume->get(x, y, z).tsdf_weight << std::endl;
                    std::cout << "color x: " << global_volume->get(x, y, z).color.x() << std::endl;
                    std::cout << "color y : " << global_volume->get(x, y, z).color.y() << std::endl;
                    std::cout << "color z: " << global_volume->get(x, y, z).color.z() << std::endl;
                    std::cout << "color w: " << global_volume->get(x, y, z).color.w() << std::endl;
                }
            }
        }
        return 0;*/

//        SurfacePrediction surface_prediction;
//        surface_prediction.predict_surface(imageProperties, global_volume);

        /*if(!surface_measurement.init(depthMap, colorMap, trajectory, trajectoryInv, depthIntrinsics))
        {
            std::cout << "Failed to read and assign data!" << std::endl;
            return -1;
        }*/
        // extract the zero iso-surface using marching cubes
/*       SimpleMesh mesh;
        for (unsigned int x = 0; x < global_volume->getDimX() - 1; x++)
        {
            for (unsigned int y = 0; y < global_volume->getDimY() - 1; y++)
            {
                for (unsigned int z = 0; z < global_volume->getDimZ() - 1; z++)
                {
                    //if (imageProperties->global_tsdf->get(x, y, z).is_occupied){
                    ProcessVolumeCell(global_volume, x, y, z, (double)0.00, &mesh);
                    //}
                }
            }
        }

        // write mesh to file
        std::stringstream ss;

        ss << "result_" << i++ << ".off";
        if (!mesh.WriteMesh(ss.str()))
        {
            std::cout << "ERROR: unable to write output file!" << std::endl;
            return -1;
        }
        //delete imageProperties;
    }

    return 0;
}

*/