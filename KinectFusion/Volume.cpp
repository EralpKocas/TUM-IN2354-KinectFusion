#include "Volume.h"
//#include "common.h"

//! Initializes an empty volume dataset.
Volume::Volume(Vector3f min_, Vector3f max_, uint dx_, uint dy_, uint dz_, uint dim)
{
	min = min_;
	max = max_;
	diag = max - min;
	dx = dx_;
	dy = dy_;
	dz = dz_;
	m_dim = dim;
	vol = NULL;

	vol = new Voxel[dx*dy*dz];

	compute_ddx_dddx();
}

Volume::~Volume()
{
	delete[] vol;
};


//! Computes spacing in x,y,z-directions.
void Volume::compute_ddx_dddx()
{
	ddx = 1.0f / (dx - 1);
	ddy = 1.0f / (dy - 1);
	ddz = 1.0f / (dz - 1);

	dddx = (max[0] - min[0]) / (dx - 1);
	dddy = (max[1] - min[1]) / (dy - 1);
	dddz = (max[2] - min[2]) / (dz - 1);

	if (dz == 1)
	{
		ddz = 0;
		dddz = 0;
	}

	diag = max - min;
}

//! Zeros out the memory
void Volume::zeroOutMemory()
{
	for (uint i1 = 0; i1 < dx*dy*dz; i1++){
		vol[i1].tsdf_distance_value = float(0);
		vol[i1].tsdf_weight = float(0);
		vol[i1].color = Vector4uc{0, 0, 0, 0};
    }
}

//! Returns the Data.
Voxel* Volume::getData()
{
	return vol;
};


//! Sets minimum extension
void Volume::SetMin(Vector3f min_)
{
	min = min_;
	diag = max - min;
}

//! Sets maximum extension
void Volume::SetMax(Vector3f max_)
{
	max = max_;
	diag = max - min;
}
