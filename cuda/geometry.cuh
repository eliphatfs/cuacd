// Geometry device functions: edge intersection, point-triangle distance,
// bbox concavity metric.

#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

#include "common.cuh"

// Signed volume of tetrahedron formed by triangle and origin.
// V = p0 . (p1 x p2) / 6
__device__ inline float signed_tet_volume(
    float p0x, float p0y, float p0z,
    float p1x, float p1y, float p1z,
    float p2x, float p2y, float p2z)
{
    float cx = p1y * p2z - p1z * p2y;
    float cy = p1z * p2x - p1x * p2z;
    float cz = p1x * p2y - p1y * p2x;
    return (p0x * cx + p0y * cy + p0z * cz) / 6.0f;
}


#endif // GEOMETRY_CUH
