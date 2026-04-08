/**
 * Standalone CoACD Hausdorff distance reference harness.
 *
 * Reads a binary mesh-pair file, computes face_hausdorff_distance using
 * the CoACD algorithm (KD-tree with 10-NN point lookup), and prints the
 * result to stdout.
 *
 * All geometric helpers are copied verbatim from CoACD/src/ (SIGGRAPH 2022).
 *
 * Compile:
 *   g++ -O2 -std=c++17 -I CoACD/src tests/ref_hausdorff.cpp \
 *       CoACD/src/sobol.cpp -o ref_hausdorff
 *
 * Usage:
 *   ./ref_hausdorff <binary_mesh_pair_file>
 *
 * Binary format (little-endian):
 *   int32  nv_a, nt_a, nv_b, nt_b
 *   float64[nv_a*3]  verts_a
 *   int32[nt_a*3]    tris_a
 *   float64[nv_b*3]  verts_b
 *   int32[nt_b*3]    tris_b
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <vector>

#include "nanoflann.hpp"
using std::string;      // sobol.h uses unqualified 'string'
#include "sobol.h"

using std::array;
using std::max;
using std::min;
using std::vector;
using namespace nanoflann;

// ---------------------------------------------------------------------------
// Types — from CoACD/src/shape.h
// ---------------------------------------------------------------------------
using vec3d = std::array<double, 3>;
using vec3i = std::array<int, 3>;
constexpr double INF_VAL = std::numeric_limits<double>::max();

// ---------------------------------------------------------------------------
// Helpers — from CoACD/src/shape.h and shape.cpp
// ---------------------------------------------------------------------------
inline vec3d CrossProduct(vec3d v, vec3d w) {
    return {v[1]*w[2] - v[2]*w[1],
            v[2]*w[0] - v[0]*w[2],
            v[0]*w[1] - v[1]*w[0]};
}

inline bool SameVectorDirection(vec3d v, vec3d w) {
    return (v[0]*w[0] + v[1]*w[1] + v[2]*w[2]) > 0;
}

vec3d CalFaceNormal(vec3d p1, vec3d p2, vec3d p3) {
    vec3d v = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
    vec3d w = {p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]};
    vec3d n = CrossProduct(v, w);
    double len = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    return {n[0]/len, n[1]/len, n[2]/len};
}

double Area(vec3d p0, vec3d p1, vec3d p2) {
    return 0.5 * sqrt(
        pow(p1[0]*p0[1] - p2[0]*p0[1] - p0[0]*p1[1] + p2[0]*p1[1] + p0[0]*p2[1] - p1[0]*p2[1], 2) +
        pow(p1[0]*p0[2] - p2[0]*p0[2] - p0[0]*p1[2] + p2[0]*p1[2] + p0[0]*p2[2] - p1[0]*p2[2], 2) +
        pow(p1[1]*p0[2] - p2[1]*p0[2] - p0[1]*p1[2] + p2[1]*p1[2] + p0[1]*p2[2] - p1[1]*p2[2], 2));
}

// Minimal Plane — from CoACD/src/shape.h
struct Plane {
    double a, b, c, d;
    short Side(vec3d p, double eps = 1e-6) {
        double res = p[0]*a + p[1]*b + p[2]*c + d;
        if (res > eps) return 1;
        else if (res < -eps) return -1;
        return 0;
    }
};

// PointCloud adapter for nanoflann — from CoACD/src/shape.h
template <typename T>
struct PointCloud {
    struct Point { T x, y, z; };
    std::vector<Point> pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

template <typename T>
void vec2PointCloud(PointCloud<T>& point, vector<vec3d> V) {
    point.pts.resize(V.size());
    for (size_t i = 0; i < V.size(); i++) {
        point.pts[i].x = V[i][0];
        point.pts[i].y = V[i][1];
        point.pts[i].z = V[i][2];
    }
}

// ---------------------------------------------------------------------------
// Model — minimal struct matching CoACD/src/model_obj.h
// ---------------------------------------------------------------------------
struct Model {
    vector<vec3d> points;
    vector<vec3i> triangles;
};

// ---------------------------------------------------------------------------
// Distance functions — verbatim from CoACD/src/hausdorff.h
// ---------------------------------------------------------------------------
double dist_point2point(vec3d pt, vec3d p) {
    return sqrt(pow(pt[0]-p[0],2) + pow(pt[1]-p[1],2) + pow(pt[2]-p[2],2));
}

double dist_point2segment(vec3d pt, vec3d s0, vec3d s1, bool flag = false) {
    vec3d BA = {pt[0]-s1[0], pt[1]-s1[1], pt[2]-s1[2]};
    vec3d BC = {s0[0]-s1[0], s0[1]-s1[1], s0[2]-s1[2]};
    double proj_dist = (BA[0]*BC[0] + BA[1]*BC[1] + BA[2]*BC[2]) /
                       sqrt(pow(BC[0],2) + pow(BC[1],2) + pow(BC[2],2));
    double valAB = sqrt(pow(BA[0],2) + pow(BA[1],2) + pow(BA[2],2));
    double valBC = sqrt(pow(BC[0],2) + pow(BC[1],2) + pow(BC[2],2));
    if (proj_dist < 0 || proj_dist > valBC)
        return INF_VAL;
    return sqrt(pow(valAB,2) - pow(proj_dist,2));
}

double dist_point2triangle(vec3d pt, vec3d tri_pt0, vec3d tri_pt1, vec3d tri_pt2, bool flag = false) {
    double _a = (tri_pt1[1]-tri_pt0[1])*(tri_pt2[2]-tri_pt0[2]) - (tri_pt1[2]-tri_pt0[2])*(tri_pt2[1]-tri_pt0[1]);
    double _b = (tri_pt1[2]-tri_pt0[2])*(tri_pt2[0]-tri_pt0[0]) - (tri_pt1[0]-tri_pt0[0])*(tri_pt2[2]-tri_pt0[2]);
    double _c = (tri_pt1[0]-tri_pt0[0])*(tri_pt2[1]-tri_pt0[1]) - (tri_pt1[1]-tri_pt0[1])*(tri_pt2[0]-tri_pt0[0]);
    double len = sqrt(_a*_a + _b*_b + _c*_c);
    double a = _a/len, b = _b/len, c = _c/len;
    double d = -(a*tri_pt0[0] + b*tri_pt0[1] + c*tri_pt0[2]);

    double dist = fabs(a*pt[0] + b*pt[1] + c*pt[2] + d) / sqrt(a*a + b*b + c*c);
    vec3d proj_pt;
    Plane p{a, b, c, d};
    short side = p.Side(pt, 1e-8);
    if (side == 1)       { proj_pt = {pt[0]-a*dist, pt[1]-b*dist, pt[2]-c*dist}; }
    else if (side == -1) { proj_pt = {pt[0]+a*dist, pt[1]+b*dist, pt[2]+c*dist}; }
    else                 { proj_pt = pt; }

    vec3d normal = CalFaceNormal(tri_pt0, tri_pt1, tri_pt2);
    vec3d AB = {tri_pt1[0]-tri_pt0[0], tri_pt1[1]-tri_pt0[1], tri_pt1[2]-tri_pt0[2]};
    vec3d BC = {tri_pt2[0]-tri_pt1[0], tri_pt2[1]-tri_pt1[1], tri_pt2[2]-tri_pt1[2]};
    vec3d CA = {tri_pt0[0]-tri_pt2[0], tri_pt0[1]-tri_pt2[1], tri_pt0[2]-tri_pt2[2]};
    vec3d AP = {proj_pt[0]-tri_pt0[0], proj_pt[1]-tri_pt0[1], proj_pt[2]-tri_pt0[2]};
    vec3d BP = {proj_pt[0]-tri_pt1[0], proj_pt[1]-tri_pt1[1], proj_pt[2]-tri_pt1[2]};
    vec3d CP = {proj_pt[0]-tri_pt2[0], proj_pt[1]-tri_pt2[1], proj_pt[2]-tri_pt2[2]};

    vec3d AB_AP = CrossProduct(AB, AP);
    vec3d BC_BP = CrossProduct(BC, BP);
    vec3d CA_CP = CrossProduct(CA, CP);

    if (SameVectorDirection(AB_AP, normal) && SameVectorDirection(BC_BP, normal) && SameVectorDirection(CA_CP, normal)) {
        return dist;
    } else {
        double d1 = dist_point2segment(pt, tri_pt0, tri_pt1);
        double d2 = dist_point2segment(pt, tri_pt1, tri_pt2);
        double d3 = dist_point2segment(pt, tri_pt2, tri_pt0);
        double d4 = dist_point2point(pt, tri_pt0);
        double d5 = dist_point2point(pt, tri_pt1);
        double d6 = dist_point2point(pt, tri_pt2);
        return min(min(min(d1,d2),d3), min(min(d4,d5),d6));
    }
}

// ---------------------------------------------------------------------------
// face_hausdorff_distance — verbatim from CoACD/src/hausdorff.h
// ---------------------------------------------------------------------------
double face_hausdorff_distance(Model& meshA, vector<vec3d>& XA, vector<int>& idA,
                               Model& meshB, vector<vec3d>& XB, vector<int>& idB) {
    int nA = XA.size();
    int nB = XB.size();
    double cmax = 0;

    PointCloud<double> cloudA, cloudB;
    vec2PointCloud(cloudA, XA);
    vec2PointCloud(cloudB, XB);

    typedef KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<double, PointCloud<double>>,
        PointCloud<double>, 3> my_kd_tree_t;

    my_kd_tree_t indexA(3, cloudA, KDTreeSingleIndexAdaptorParams(10));
    my_kd_tree_t indexB(3, cloudB, KDTreeSingleIndexAdaptorParams(10));
    indexA.buildIndex();
    indexB.buildIndex();

    // B -> A direction
    for (int i = 0; i < nB; i++) {
        size_t num_results = 10;
        double query_pt[3] = {XB[i][0], XB[i][1], XB[i][2]};
        std::vector<size_t> ret_index(num_results);
        std::vector<double> out_dist_sqr(num_results);
        num_results = indexA.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);

        double cmin = INF_VAL;
        for (int j = 0; j < (int)num_results; j++) {
            double distance = dist_point2triangle(XB[i],
                meshA.points[meshA.triangles[idA[ret_index[j]]][0]],
                meshA.points[meshA.triangles[idA[ret_index[j]]][1]],
                meshA.points[meshA.triangles[idA[ret_index[j]]][2]]);
            if (distance < cmin) {
                cmin = distance;
                if (cmin < 1e-14) break;
            }
        }
        if (cmin > 10) cmin = sqrt(out_dist_sqr[0]);
        if (cmin > cmax && INF_VAL > cmin) cmax = cmin;
    }

    // A -> B direction
    for (int i = 0; i < nA; i++) {
        size_t num_results = 10;
        double query_pt[3] = {XA[i][0], XA[i][1], XA[i][2]};
        std::vector<size_t> ret_index(num_results);
        std::vector<double> out_dist_sqr(num_results);
        num_results = indexB.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);

        double cmin = INF_VAL;
        for (int j = 0; j < (int)num_results; j++) {
            double distance = dist_point2triangle(XA[i],
                meshB.points[meshB.triangles[idB[ret_index[j]]][0]],
                meshB.points[meshB.triangles[idB[ret_index[j]]][1]],
                meshB.points[meshB.triangles[idB[ret_index[j]]][2]]);
            if (distance < cmin) {
                cmin = distance;
                if (cmin < 1e-14) break;
            }
        }
        if (cmin > 10) cmin = sqrt(out_dist_sqr[0]);
        if (cmin > cmax && INF_VAL > cmin) cmax = cmin;
    }

    return cmax;
}

// ---------------------------------------------------------------------------
// ExtractPointSet — adapted from CoACD/src/model_obj.cpp:346-403
// ---------------------------------------------------------------------------
static std::mt19937 rng_engine;

void ExtractPointSet(Model& mesh, vector<vec3d>& samples, vector<int>& sample_tri_ids,
                     unsigned int seed, size_t resolution) {
    if (resolution == 0) return;
    rng_engine.seed(seed);

    double aObj = 0;
    for (int i = 0; i < (int)mesh.triangles.size(); i++)
        aObj += Area(mesh.points[mesh.triangles[i][0]],
                     mesh.points[mesh.triangles[i][1]],
                     mesh.points[mesh.triangles[i][2]]);

    for (int i = 0; i < (int)mesh.triangles.size(); i++) {
        double area = Area(mesh.points[mesh.triangles[i][0]],
                           mesh.points[mesh.triangles[i][1]],
                           mesh.points[mesh.triangles[i][2]]);
        int N;
        if ((size_t)mesh.triangles.size() > resolution && resolution)
            N = max(int(i % ((int)mesh.triangles.size() / (int)resolution) == 0),
                    int(resolution / aObj * area));
        else
            N = max(int(i % 2 == 0), int(resolution / aObj * area));

        std::uniform_int_distribution<int> seeder(0, 1000);
        int sobol_seed = seeder(rng_engine);
        float r[2];
        for (int k = 0; k < N; k++) {
            double a, b;
            if (k % 3 == 0) {
                std::uniform_real_distribution<double> uniform(0.0, 1.0);
                a = uniform(rng_engine);
                b = uniform(rng_engine);
            } else {
                i4_sobol(2, &sobol_seed, r);
                a = r[0];
                b = r[1];
            }

            vec3d v;
            v[0] = (1-sqrt(a)) * mesh.points[mesh.triangles[i][0]][0]
                 + (sqrt(a)*(1-b)) * mesh.points[mesh.triangles[i][1]][0]
                 + b*sqrt(a) * mesh.points[mesh.triangles[i][2]][0];
            v[1] = (1-sqrt(a)) * mesh.points[mesh.triangles[i][0]][1]
                 + (sqrt(a)*(1-b)) * mesh.points[mesh.triangles[i][1]][1]
                 + b*sqrt(a) * mesh.points[mesh.triangles[i][2]][1];
            v[2] = (1-sqrt(a)) * mesh.points[mesh.triangles[i][0]][2]
                 + (sqrt(a)*(1-b)) * mesh.points[mesh.triangles[i][1]][2]
                 + b*sqrt(a) * mesh.points[mesh.triangles[i][2]][2];
            samples.push_back(v);
            sample_tri_ids.push_back(i);
        }
    }
}

// ---------------------------------------------------------------------------
// Binary I/O
// ---------------------------------------------------------------------------
static bool read_mesh_pair(const char* path,
                           Model& meshA, Model& meshB) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }

    int32_t nv_a, nt_a, nv_b, nt_b;
    fread(&nv_a, 4, 1, f);
    fread(&nt_a, 4, 1, f);
    fread(&nv_b, 4, 1, f);
    fread(&nt_b, 4, 1, f);

    // Read verts_a (float64)
    meshA.points.resize(nv_a);
    for (int i = 0; i < nv_a; i++) {
        double xyz[3];
        fread(xyz, sizeof(double), 3, f);
        meshA.points[i] = {xyz[0], xyz[1], xyz[2]};
    }
    // Read tris_a (int32)
    meshA.triangles.resize(nt_a);
    for (int i = 0; i < nt_a; i++) {
        int32_t tri[3];
        fread(tri, sizeof(int32_t), 3, f);
        meshA.triangles[i] = {tri[0], tri[1], tri[2]};
    }
    // Read verts_b (float64)
    meshB.points.resize(nv_b);
    for (int i = 0; i < nv_b; i++) {
        double xyz[3];
        fread(xyz, sizeof(double), 3, f);
        meshB.points[i] = {xyz[0], xyz[1], xyz[2]};
    }
    // Read tris_b (int32)
    meshB.triangles.resize(nt_b);
    for (int i = 0; i < nt_b; i++) {
        int32_t tri[3];
        fread(tri, sizeof(int32_t), 3, f);
        meshB.triangles[i] = {tri[0], tri[1], tri[2]};
    }
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <binary_mesh_pair_file>\n", argv[0]);
        return 1;
    }

    Model meshA, meshB;
    if (!read_mesh_pair(argv[1], meshA, meshB))
        return 1;

    unsigned int seed = 1234;
    size_t resolution = 2000;

    vector<vec3d> samplesA, samplesB;
    vector<int> idsA, idsB;
    ExtractPointSet(meshA, samplesA, idsA, seed, resolution);
    ExtractPointSet(meshB, samplesB, idsB, seed, resolution);

    fprintf(stderr, "meshA: %zu verts, %zu tris, %zu samples\n",
            meshA.points.size(), meshA.triangles.size(), samplesA.size());
    fprintf(stderr, "meshB: %zu verts, %zu tris, %zu samples\n",
            meshB.points.size(), meshB.triangles.size(), samplesB.size());

    double h = face_hausdorff_distance(meshA, samplesA, idsA, meshB, samplesB, idsB);
    printf("%.17g\n", h);
    return 0;
}
