/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  FULL NAME <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
        : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::decomposeTree(const Vec3_t<float> &pos,
                                        const ParametricScalarField &field,
                                        const unsigned &gridSize)
{
    unsigned totalTriangles = 0;

    unsigned gridSize_ = gridSize / 2;
    auto gridSizeF = float(gridSize_);

    float x = (pos.x + gridSizeF) * mGridResolution;
    float y = (pos.y + gridSizeF) * mGridResolution;
    float z = (pos.z + gridSizeF) * mGridResolution;

    Vec3_t<float> midPoint(x, y, z);
    float f = field.getIsoLevel() + ((sqrt(3) / 2) * (gridSize * mGridResolution));

    /* check if the block is empty */
    if (evaluateFieldAt(midPoint, field) > f) {
        return 0;
    }

    /* cut off */
    if (gridSize <= 1) {
        return buildCube(pos, field);
    }

    for (Vec3_t<float> point : sc_vertexNormPos) {
        #pragma omp task default(none) \
            shared(pos, field, gridSize_, gridSizeF, totalTriangles) \
            firstprivate(point)
        {
            Vec3_t<float> cubeOffset(pos.x + point.x * gridSizeF,
                                     pos.y + point.y * gridSizeF,
                                     pos.z + point.z * gridSizeF);

            unsigned triangles = decomposeTree(cubeOffset, field, gridSize_);
            #pragma omp critical
            totalTriangles += triangles;
        }
    }

    #pragma omp taskwait
    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    unsigned trianglesCount;

    #pragma omp parallel default(none) shared(trianglesCount, field)
    #pragma omp single nowait
    trianglesCount = decomposeTree(Vec3_t<float>(), field, mGridSize);

    return trianglesCount;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        value = std::min(value, distanceSquared);
    }

    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical
    mTriangles.push_back(triangle);
}