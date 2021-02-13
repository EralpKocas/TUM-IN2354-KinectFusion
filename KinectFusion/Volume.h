#pragma once

#ifndef VOLUME_H
#define VOLUME_H

#include <limits>
#include "Eigen.h"
//#include "common.h"
typedef unsigned int uint;

struct Voxel
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // position stored as 4 floats (4th component is supposed to be 1.0)
    float tsdf_distance_value;
    // color stored as 4 unsigned char
    int tsdf_weight;
    Vector4uc color;
    bool is_occupied = false;
};

//! A regular volume dataset
class Volume
{
public:

	//! Initializes an empty volume dataset.
	Volume(Vector3f min_, Vector3f max_, uint dx_ = 10, uint dy_ = 10, uint dz_ = 10, uint dim = 1);

	~Volume();

	inline void computeMinMaxValues(double& minVal, double& maxVal) const
	{
		minVal = std::numeric_limits<double>::max();
		maxVal = -minVal;
		for (uint i1 = 0; i1 < dx*dy*dz; i1++)
		{
			if (minVal > vol[i1].tsdf_distance_value) minVal = vol[i1].tsdf_distance_value;
			if (maxVal < vol[i1].tsdf_distance_value) maxVal = vol[i1].tsdf_distance_value;
		}
	}

	//! Computes spacing in x,y,z-directions.
	void compute_ddx_dddx();

	//! Zeros out the memory
	void zeroOutMemory();

	//! Set the value at i.
	inline void set(uint i, double val)
	{
		if (val > maxValue)
			maxValue = val;

		if (val < minValue)
			minValue = val;

		vol[i].tsdf_distance_value = (float) val;
	}

	//! Set the value at (x_, y_, z_).
	inline void set(uint x_, uint y_, uint z_, Voxel val)
	{
		vol[getPosFromTuple(x_, y_, z_)] = val;
	};

	//! Get the value at (x_, y_, z_).
	inline Voxel get(uint i) const
	{
		return vol[i];
	};

	//! Get the value at (x_, y_, z_).
	inline Voxel get(uint x_, uint y_, uint z_) const
	{
		return vol[getPosFromTuple(x_, y_, z_)];
	};

	//! Get the value at (pos.x, pos.y, pos.z).
	inline Voxel get(const Vector3i& pos_) const
	{
		return(get(pos_[0], pos_[1], pos_[2]));
	}

	//! Returns the cartesian x-coordinates of node (i,..).
	inline double posX(int i) const
	{
		return min[0] + diag[0] * (double(i)*ddx);
	}

	//! Returns the cartesian y-coordinates of node (..,i,..).
	inline double posY(int i) const
	{
		return min[1] + diag[1] * (double(i)*ddy);
	}

	//! Returns the cartesian z-coordinates of node (..,i).
	inline double posZ(int i) const
	{
		return min[2] + diag[2] * (double(i)*ddz);
	}

	//! Returns the cartesian coordinates of node (i,j,k).
	inline Vector3f pos(int i, int j, int k) const
	{
		Vector3f coord(0, 0, 0);

		coord[0] = min[0] + (max[0] - min[0])*(double(i)*ddx);
		coord[1] = min[1] + (max[1] - min[1])*(double(j)*ddy);
		coord[2] = min[2] + (max[2] - min[2])*(double(k)*ddz);

		return coord;
	}

	//! Returns the corresponding node of given cartesian coordinates.
	inline Vector3f compute_grid(Vector3f p)
    {
        return Vector3f(((p[0] - min[0]) / (max[0] - min[0])) / ddx,
                        ((p[1] - min[1]) / (max[1] - min[1])) / ddy,
                        ((p[2] - min[2]) / (max[2] - min[2])) / ddz
                );
    }

    inline Voxel set_occupied(Vector3f p){

	    Voxel voxel = get(((p[0] - min[0]) / (max[0] - min[0])) / ddx,
            ((p[1] - min[1]) / (max[1] - min[1])) / ddy,
            ((p[2] - min[2]) / (max[2] - min[2])) / ddz);
        Voxel curr_voxel;
        curr_voxel.tsdf_distance_value = voxel.tsdf_distance_value;
        curr_voxel.tsdf_weight = voxel.tsdf_weight;
        curr_voxel.color = voxel.color;
        curr_voxel.is_occupied = true;
        return curr_voxel;
	}

	//! Returns the Data.
	Voxel* getData();

	//! Sets all entries in the volume to '0'
	void clean();

	//! Returns number of cells in x-dir.
	inline uint getDimX() const { return dx; }

	//! Returns number of cells in y-dir.
	inline uint getDimY() const { return dy; }

	//! Returns number of cells in z-dir.
	inline uint getDimZ() const { return dz; }

	inline Vector3f getMin() { return min; }
	inline Vector3f getMax() { return max; }

	//! Sets minimum extension
	void SetMin(Vector3f min_);

	//! Sets maximum extension
	void SetMax(Vector3f max_);

	inline uint getPosFromTuple(int x, int y, int z) const
	{
		return x*dy*dz + y*dz + z;
	}

    //! VARIABLES

	//! Lower left and Upper right corner.
	Vector3f min, max;

	//! max-min
	Vector3f diag;

	double ddx, ddy, ddz;
	double dddx, dddy, dddz;

	//! Number of cells in x, y and z-direction.
	uint dx, dy, dz;

	Voxel* vol;
	double maxValue, minValue;
	uint m_dim;

private:

	//! x,y,z access to vol*
	inline Voxel vol_access(int x, int y, int z) const
	{
		return vol[getPosFromTuple(x, y, z)];
	}
};

#endif // VOLUME_H
