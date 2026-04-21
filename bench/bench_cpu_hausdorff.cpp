// bench_cpu_hausdorff.cpp -- Standalone CPU Hausdorff benchmark wrapper.
//
// Plugs into hausdorff_bench.cu via extern "C" C linkage.
//
// Key trick: we include CoACD's hausdorff.h for the face_hausdorff_distance
// algorithm, but we provide our own stub for Model::Model() and a minimal
// surface sampler so we do NOT need to compile model_obj.cpp (which pulls in
// process.h -> preprocess.h -> openvdb).  We only link shape.cpp for
// CalFaceNormal (used by dist_point2triangle).
// ============================================================================

#define DISABLE_SPDLOG

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

// Minimal stub for coacd::Model so we can use face_hausdorff_distance without
// pulling in model_obj.h (which transitively includes openvdb headers).
// This must match the subset of Model used by hausdorff.h.
#include "shape.h"
namespace coacd {
    class Model {
    public:
        std::vector<vec3d> points;
        std::vector<vec3i> triangles;
        Model();
    };
    Model::Model() {}
}

#include "hausdorff.h"

// ============================================================================
// Surface sampler (area-weighted uniform random, resolution samples total).
// ============================================================================
static void sample_surface(const coacd::Model& mesh,
                           std::vector<coacd::vec3d>& samples,
                           std::vector<int>& tri_ids,
                           unsigned int seed,
                           unsigned int resolution)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d01(0.0, 1.0);

    size_t nt = mesh.triangles.size();
    if (nt == 0) return;

    std::vector<double> areas(nt);
    double total_area = 0.0;
    for (size_t i = 0; i < nt; ++i) {
        const auto& t = mesh.triangles[i];
        areas[i] = coacd::Area(mesh.points[t[0]], mesh.points[t[1]], mesh.points[t[2]]);
        total_area += areas[i];
    }
    if (total_area < 1e-20) return;

    samples.reserve(resolution);
    tri_ids.reserve(resolution);

    for (unsigned int s = 0; s < resolution; ++s) {
        double r = d01(rng) * total_area;
        size_t tri = 0;
        double cum = 0.0;
        for (; tri < nt; ++tri) {
            cum += areas[tri];
            if (cum >= r) break;
        }
        if (tri >= nt) tri = nt - 1;

        double u = d01(rng);
        double v = d01(rng);
        double su = std::sqrt(u);
        double b0 = 1.0 - su;
        double b1 = su * (1.0 - v);
        double b2 = su * v;

        const auto& p0 = mesh.points[mesh.triangles[tri][0]];
        const auto& p1 = mesh.points[mesh.triangles[tri][1]];
        const auto& p2 = mesh.points[mesh.triangles[tri][2]];

        coacd::vec3d p;
        p[0] = static_cast<double>(b0 * p0[0] + b1 * p1[0] + b2 * p2[0]);
        p[1] = static_cast<double>(b0 * p0[1] + b1 * p1[1] + b2 * p2[1]);
        p[2] = static_cast<double>(b0 * p0[2] + b1 * p1[2] + b2 * p2[2]);

        samples.push_back(p);
        tri_ids.push_back(static_cast<int>(tri));
    }
}

// ============================================================================
// C API for the benchmark harness
// ============================================================================

static double run_once(const float* verts_a, const int* tris_a, int nv_a, int nt_a,
                       const float* verts_b, const int* tris_b, int nv_b, int nt_b,
                       unsigned int resolution, unsigned int seed)
{
    coacd::Model ma, mb;
    ma.points.resize(static_cast<size_t>(nv_a));
    for (int i = 0; i < nv_a; ++i) {
        ma.points[i][0] = verts_a[i * 3 + 0];
        ma.points[i][1] = verts_a[i * 3 + 1];
        ma.points[i][2] = verts_a[i * 3 + 2];
    }
    ma.triangles.resize(static_cast<size_t>(nt_a));
    for (int i = 0; i < nt_a; ++i) {
        ma.triangles[i][0] = tris_a[i * 3 + 0];
        ma.triangles[i][1] = tris_a[i * 3 + 1];
        ma.triangles[i][2] = tris_a[i * 3 + 2];
    }

    mb.points.resize(static_cast<size_t>(nv_b));
    for (int i = 0; i < nv_b; ++i) {
        mb.points[i][0] = verts_b[i * 3 + 0];
        mb.points[i][1] = verts_b[i * 3 + 1];
        mb.points[i][2] = verts_b[i * 3 + 2];
    }
    mb.triangles.resize(static_cast<size_t>(nt_b));
    for (int i = 0; i < nt_b; ++i) {
        mb.triangles[i][0] = tris_b[i * 3 + 0];
        mb.triangles[i][1] = tris_b[i * 3 + 1];
        mb.triangles[i][2] = tris_b[i * 3 + 2];
    }

    std::vector<coacd::vec3d> sa, sb;
    std::vector<int> ida, idb;
    sample_surface(ma, sa, ida, seed, resolution);
    sample_surface(mb, sb, idb, seed, resolution);

    return coacd::face_hausdorff_distance(ma, sa, ida, mb, sb, idb);
}

extern "C" double bench_cpu_hausdorff_single(const float* verts_a, const int* tris_a,
                                                int nv_a, int nt_a,
                                                const float* verts_b, const int* tris_b,
                                                int nv_b, int nt_b,
                                                unsigned int resolution,
                                                unsigned int seed)
{
    return run_once(verts_a, tris_a, nv_a, nt_a,
                    verts_b, tris_b, nv_b, nt_b,
                    resolution, seed);
}

extern "C" double bench_cpu_hausdorff_batch(const float* verts_a, const int* tris_a,
                                               int nv_a, int nt_a,
                                               const float* verts_b, const int* tris_b,
                                               int nv_b, int nt_b,
                                               int batch,
                                               unsigned int resolution,
                                               unsigned int seed)
{
    double sum = 0.0;
    for (int i = 0; i < batch; ++i) {
        sum += run_once(verts_a, tris_a, nv_a, nt_a,
                        verts_b, tris_b, nv_b, nt_b,
                        resolution, seed);
    }
    return sum / batch;
}
