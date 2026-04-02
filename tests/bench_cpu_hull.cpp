/*
 * bench_cpu_hull.cpp
 *
 * Standalone benchmark for btConvexHullComputer (Preparata-Hong D&C hull, CPU).
 * Mirrors the batch configs and distributions from tests/test_hull.py / docs/arena_sweep.md:
 *
 *   Configs:  10000x20pts, 1000x200pts, 100x2000pts, 10x20000pts
 *   Dists:    sphere_shell, cube_interior, gaussian
 *
 * Build:
 *   g++ -O2 -std=c++17 -I CoACD/src/btConvexHull \
 *       tests/bench_cpu_hull.cpp \
 *       CoACD/src/btConvexHull/btConvexHullComputer.cpp \
 *       CoACD/src/btConvexHull/btAlignedAllocator.cpp \
 *       -o bench_cpu_hull
 *
 * Run:
 *   ./bench_cpu_hull
 */

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "btConvexHullComputer.h"
#include "btVector3.h"

// ---------------------------------------------------------------------------
// Simple PCG-based RNG seeded deterministically (mirrors numpy default_rng(0))
// We use std::mt19937 seeded with 0 to match spirit of seed=0.
// ---------------------------------------------------------------------------
static std::mt19937_64 rng_state;

static void rng_seed(uint64_t seed) { rng_state.seed(seed); }

static double rng_normal() {
    static std::normal_distribution<double> nd(0.0, 1.0);
    return nd(rng_state);
}

static double rng_uniform() {
    static std::uniform_real_distribution<double> ud(0.0, 1.0);
    return ud(rng_state);
}

// ---------------------------------------------------------------------------
// Point distributions
// ---------------------------------------------------------------------------

using Pts = std::vector<std::array<double, 3>>;

static Pts make_sphere_shell(int n) {
    Pts pts(n);
    for (int i = 0; i < n; ++i) {
        double x = rng_normal(), y = rng_normal(), z = rng_normal();
        double r = std::sqrt(x*x + y*y + z*z);
        if (r < 1e-12) r = 1.0;
        pts[i] = {x/r, y/r, z/r};
    }
    return pts;
}

static Pts make_cube_interior(int n) {
    Pts pts(n);
    for (int i = 0; i < n; ++i)
        pts[i] = {rng_uniform() - 0.5, rng_uniform() - 0.5, rng_uniform() - 0.5};
    return pts;
}

static Pts make_gaussian(int n) {
    Pts pts(n);
    for (int i = 0; i < n; ++i)
        pts[i] = {rng_normal(), rng_normal(), rng_normal()};
    return pts;
}

// ---------------------------------------------------------------------------
// Divergence-theorem volume from btConvexHullComputer output (same formula
// as mesh_volume_cpu in test_hull.py).
// ---------------------------------------------------------------------------
static double hull_volume(const btConvexHullComputer& ch) {
    double vol = 0.0;
    int nf = ch.faces.size();
    for (int f = 0; f < nf; ++f) {
        const btConvexHullComputer::Edge* e0 = &ch.edges[ch.faces[f]];
        int a = e0->getSourceVertex();
        int b = e0->getTargetVertex();
        const btConvexHullComputer::Edge* e = e0->getNextEdgeOfFace();
        int c = e->getTargetVertex();
        // triangulate fan
        while (c != a) {
            const auto& va = ch.vertices[a];
            const auto& vb = ch.vertices[b];
            const auto& vc = ch.vertices[c];
            // signed tet volume: (va × vb) · vc / 6
            vol += (va.getX()*(vb.getY()*vc.getZ() - vb.getZ()*vc.getY())
                  - va.getY()*(vb.getX()*vc.getZ() - vb.getZ()*vc.getX())
                  + va.getZ()*(vb.getX()*vc.getY() - vb.getY()*vc.getX()));
            e = e->getNextEdgeOfFace();
            b = c;
            c = e->getTargetVertex();
        }
    }
    return std::abs(vol) / 6.0;
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;
static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------
int main() {
    struct Config { int n_hulls, n_pts; const char* name; };
    static const Config configs[] = {
        {10000,    20, "10000x20pts"},
        { 1000,   200, "1000x200pts"},
        {  100,  2000, "100x2000pts"},
        {   10, 20000, "10x20000pts"},
    };

    struct Dist {
        const char* name;
        Pts (*fn)(int);
    };
    static const Dist dists[] = {
        {"sphere_shell",  make_sphere_shell},
        {"cube_interior", make_cube_interior},
        {"gaussian",      make_gaussian},
    };

    printf("\n%-20s %-16s %9s %10s\n", "Config", "Distribution", "CPU ms", "mean vol");
    printf("%s\n", std::string(60, '-').c_str());

    for (auto& cfg : configs) {
        for (auto& d : dists) {
            // regenerate with fixed seed for reproducibility
            rng_seed(0);

            // pre-generate all point clouds
            std::vector<Pts> batch(cfg.n_hulls);
            for (int i = 0; i < cfg.n_hulls; ++i)
                batch[i] = d.fn(cfg.n_pts);

            // warm up (first hull)
            {
                btConvexHullComputer ch;
                ch.compute(batch[0], -1.0, -1.0);
            }

            auto t0 = Clock::now();
            double vol_sum = 0.0;
            for (int i = 0; i < cfg.n_hulls; ++i) {
                btConvexHullComputer ch;
                ch.compute(batch[i], -1.0, -1.0);
                vol_sum += hull_volume(ch);
            }
            double ms = elapsed_ms(t0);

            printf("%-20s %-16s %9.1f %10.4f\n",
                   cfg.name, d.name, ms, vol_sum / cfg.n_hulls);
        }
    }
    printf("\n");
    return 0;
}
