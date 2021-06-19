#include <iostream>
#include <fstream>
#include <string>

#include <math.h>
#include "common_functions.h"
#include "VirtualSensor_freiburg.h"

__global__ // This keyword means the code runs on the GPU.
void add(int n, float *x, float *y)
    {
    // At each index, add x to y.
    for (int i = 0; i < n; i++)
        {
        y[i] = x[i] + y[i];
        }
    }

int main(void)
    {
    int N = 1000000;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // Initialize our x and y arrays with some floats.
    for (int i = 0; i < N; i++)
        {
        x[i] = 1.0f;
        y[i] = 2.0f;
        }

    // Run the function on using the GPU.
    add<<<1, 1>>>(N, x, y); // Notice the brackets.

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        {
        maxError = fmax(maxError, fabs(y[i]-3.0f));
        }
    std::cout << "Max error: " << maxError << std::endl;
    // Free memory
    cudaFree(x);
    cudaFree(y);

    // Make sure this path points to the data folder
    //std::string filenameIn = "/Users/beyzatugcebilgic/Desktop/TUM-IN2354-KinectFusion/KinectFusion/data/rgbd_dataset_freiburg1_xyz/";
    //std::string filenameIn = "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/data/rgbd_dataset_freiburg1_xyz/";
    std::string filenameIn = "/media/eralpkocas/hdd/TUM/3D_Scanning/data/rgbd_dataset_freiburg1_xyz/";
    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor_freiburg sensor;
    if (!sensor.init(filenameIn))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    while (sensor.processNextFrame()) {

    }

    return 0;
    }