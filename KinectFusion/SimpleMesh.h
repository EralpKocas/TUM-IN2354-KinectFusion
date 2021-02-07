#pragma once

#ifndef SIMPLE_MESH_H
#define SIMPLE_MESH_H

#include <iostream>
#include <fstream>

#include "Eigen.h"

//typedef Eigen::Vector3f Vertex;

struct Vertex{
    Eigen::Vector3f vertex_pos;
    Vector4uc vertex_color;
};

struct Triangle
{
	unsigned int idx0;
	unsigned int idx1;
	unsigned int idx2;
	Triangle(unsigned int _idx0, unsigned int _idx1, unsigned int _idx2) :
		idx0(_idx0), idx1(_idx1), idx2(_idx2)
	{}
};

class SimpleMesh
{
public:

	void Clear()
	{
		m_vertices.clear();
		m_triangles.clear();
	}

	unsigned int AddVertex(Vertex& vertex)
	{
		unsigned int vId = (unsigned int)m_vertices.size();
		m_vertices.push_back(vertex);
		return vId;
	}

	unsigned int AddFace(unsigned int idx0, unsigned int idx1, unsigned int idx2)
	{
		unsigned int fId = (unsigned int)m_triangles.size();
		Triangle triangle(idx0, idx1, idx2);
		m_triangles.push_back(triangle);
		return fId;
	}

	std::vector<Vertex>& GetVertices()
	{
		return m_vertices;
	}

	std::vector<Triangle>& GetTriangles()
	{
		return m_triangles;
	}

    bool isDistValid(Vertex p1, Vertex p2, Vertex p3, float edgeThreshold){
        float x1 = p1.vertex_pos.x();
        float y1 = p1.vertex_pos.y();
        float z1 = p1.vertex_pos.z();

        float x2 = p2.vertex_pos.x();
        float y2 = p2.vertex_pos.y();
        float z2 = p2.vertex_pos.z();

        float x3 = p3.vertex_pos.x();
        float y3 = p3.vertex_pos.y();
        float z3 = p3.vertex_pos.z();

        float dist1 = pow((pow((x1-x2), 2)+ pow((y1-y2), 2) + pow((z1-z2), 2)), (0.5));
        float dist2 = pow((pow((x1-x3), 2)+ pow((y1-y3), 2) + pow((z1-z3), 2)), (0.5));
        float dist3 = pow((pow((x2-x3), 2)+ pow((y2-y3), 2) + pow((z2-z3), 2)), (0.5));

        if (dist1 < edgeThreshold and dist2 < edgeThreshold and dist3 < edgeThreshold) return TRUE;
        return FALSE;
    }

    void create_triangles()
    {
        int depthHeight = 480;
        int depthWidth = 640;
        float edgeThreshold = 0.01f;
        for (unsigned int i = 0; i < depthHeight - 1; i++) {
            for (unsigned int j = 0; j < depthWidth - 1; j++) {
                unsigned int i0 = i * depthWidth + j;
                unsigned int i1 = (i + 1) * depthWidth + j;
                unsigned int i2 = i * depthWidth + j + 1;
                unsigned int i3 = (i + 1) * depthWidth + j + 1;

                bool valid0 = m_vertices[i0].vertex_pos.allFinite();
                bool valid1 = m_vertices[i1].vertex_pos.allFinite();
                bool valid2 = m_vertices[i2].vertex_pos.allFinite();
                bool valid3 = m_vertices[i3].vertex_pos.allFinite();

                if (valid0 && valid1 && valid2) {
                    float d0 = (m_vertices[i0].vertex_pos - m_vertices[i1].vertex_pos).norm();
                    float d1 = (m_vertices[i0].vertex_pos - m_vertices[i2].vertex_pos).norm();
                    float d2 = (m_vertices[i1].vertex_pos - m_vertices[i2].vertex_pos).norm();

                    if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2)
                        AddFace(i0, i1, i2);
                }
                if (valid1 && valid2 && valid3) {
                    float d0 = (m_vertices[i3].vertex_pos - m_vertices[i1].vertex_pos).norm();
                    float d1 = (m_vertices[i3].vertex_pos - m_vertices[i2].vertex_pos).norm();
                    float d2 = (m_vertices[i1].vertex_pos - m_vertices[i2].vertex_pos).norm();
                    if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2)
                        AddFace(i1, i3, i2);
                }
            }
        }

    }

	bool WriteMesh(const std::string& filename)
	{
		// Write off file
		std::ofstream outFile(filename);
		if (!outFile.is_open()) return false;
        //create_triangles();
		// write header
		outFile << "COFF" << std::endl;
		outFile << m_vertices.size() << " " << m_triangles.size() << " 0" << std::endl;

		// save vertices
		for (unsigned int i = 0; i<m_vertices.size(); i++)
		{
		    if(m_vertices[i].vertex_pos.allFinite())
			    outFile << m_vertices[i].vertex_pos.x() << " " << m_vertices[i].vertex_pos.y() << " " << m_vertices[i].vertex_pos.z()  << " " << +m_vertices[i].vertex_color[0] << " " << +m_vertices[i].vertex_color[1]<< " " << +m_vertices[i].vertex_color[2]
                                                                                                  << " " << +m_vertices[i].vertex_color[3]<< std::endl;
		    else
                outFile << "0.0 0.0 0.0 0 0 0 0" << std::endl;
        }

		// save faces
		for (unsigned int i = 0; i<m_triangles.size(); i++)
		{
			outFile << "3 " << m_triangles[i].idx0 << " " << m_triangles[i].idx1 << " " << m_triangles[i].idx2 << std::endl;
		}

		/*int nVertices = m_vertices.size();
		float edgeThreshold = 0.01f;
		int width = 640;
        for(int i=0; i < nVertices; i++){
            if (i != nVertices - 1 and (m_vertices[i].vertex_pos.x() != MINF && !isnan(m_vertices[i].vertex_pos.x()))){
                if (i % width == width-1 or i >= nVertices - width) continue;

                int corner_2 = i + width;
                int corner_3 = i + 1;
                bool distTrue = isDistValid(m_vertices[i], m_vertices[corner_2], m_vertices[corner_3], edgeThreshold);
                if ((m_vertices[corner_2].vertex_pos.x() != MINF && !isnan(m_vertices[i].vertex_pos.x()))
                and (m_vertices[corner_3].vertex_pos.x() != MINF && !isnan(m_vertices[i].vertex_pos.x())) and distTrue){
                    outFile << "3 " << i << " " << corner_2 << " " << corner_3 << std::endl;
                } else continue;

                int corner_4 = corner_2 + 1;
                distTrue = isDistValid(m_vertices[corner_2], m_vertices[corner_3], m_vertices[corner_4], edgeThreshold);
                if ((m_vertices[corner_4].vertex_pos.x() != MINF && !isnan(m_vertices[i].vertex_pos.x())) and distTrue)
                    outFile << "3 " << corner_2 << " " << corner_4 << " " << corner_3 << std::endl;
            }
        }*/

		// close file
		outFile.close();

		return true;
	}

private:
	std::vector<Vertex> m_vertices;
	std::vector<Triangle> m_triangles;
};

class PointCloud
{
public:
	bool ReadFromFile(const std::string& filename)
	{
		std::ifstream is(filename, std::ios::in | std::ios::binary);
		if (!is.is_open())
		{
			std::cout << "ERROR: unable to read input file!" << std::endl;
			return false;
		}

		char nBytes;
		is.read(&nBytes, sizeof(char));

		unsigned int n;
		is.read((char*)&n, sizeof(unsigned int));

		if (nBytes == sizeof(float))
		{
			float* ps = new float[3 * n];

			is.read((char*)ps, 3 * sizeof(float)*n);

			for (unsigned int i = 0; i < n; i++)
			{
				Eigen::Vector3f p(ps[3 * i + 0], ps[3 * i + 1], ps[3 * i + 2]);
				m_points.push_back(p);
			}

			is.read((char*)ps, 3 * sizeof(float)*n);
			for (unsigned int i = 0; i < n; i++)
			{
				Eigen::Vector3f p(ps[3 * i + 0], ps[3 * i + 1], ps[3 * i + 2]);
				m_normals.push_back(p);
			}

			delete ps;
		}
		else
		{
			double* ps = new double[3 * n];

			is.read((char*)ps, 3 * sizeof(double)*n);

			for (unsigned int i = 0; i < n; i++)
			{
				Eigen::Vector3f p((float)ps[3 * i + 0], (float)ps[3 * i + 1], (float)ps[3 * i + 2]);
				m_points.push_back(p);
			}

			is.read((char*)ps, 3 * sizeof(double)*n);

			for (unsigned int i = 0; i < n; i++)
			{
				Eigen::Vector3f p((float)ps[3 * i + 0], (float)ps[3 * i + 1], (float)ps[3 * i + 2]);
				m_normals.push_back(p);
			}

			delete ps;
		}


		//std::ofstream file("pointcloud.off");
		//file << "OFF" << std::endl;
		//file << m_points.size() << " 0 0" << std::endl;
		//for(unsigned int i=0; i<m_points.size(); ++i)
		//	file << m_points[i].x() << " " << m_points[i].y() << " " << m_points[i].z() << std::endl;
		//file.close();

		return true;
	}

	std::vector<Eigen::Vector3f>& GetPoints()
	{
		return m_points;
	}

	std::vector<Eigen::Vector3f>& GetNormals()
	{
		return m_normals;
	}

	unsigned int GetClosestPoint(Eigen::Vector3f& p)
	{
		unsigned int idx = 0;

		float min_dist = std::numeric_limits<float>::max();
		for (unsigned int i = 0; i < m_points.size(); ++i)
		{
			float dist = (p - m_points[i]).norm();
			if (min_dist > dist)
			{
				idx = i;
				min_dist = dist;
			}
		}

		return idx;
	}

private:
	std::vector<Eigen::Vector3f> m_points;
	std::vector<Eigen::Vector3f> m_normals;

};

#endif // SIMPLE_MESH_H
