// ============================================================================
// hull_bench.cu -- standalone perf bench for convex hull: CPU vs GPU.
//
// Compares:
//   host:   btConvexHullComputer (Preparata-Hong D&C, from CoACD/Bullet)
//   device: hull_dandc_warp_mesh  -- exact D&C hull (1 warp per hull)
//   device: kdop_hull_block       -- k-DOP prefilter + D&C (1 block/32 threads)
//
// Distributions: gaussian N(0,1), uniform cube [-0.5, 0.5]
// Batch:  256
// N pts:  512, 2048, 16384, 65536
//
// Build (from bench/):
//   make -f Makefile.hull            # default ARCH=89 (RTX 4090)
//   make -f Makefile.hull verbose    # with ptxas -v
// Run:
//   ./hull_bench > results_hull.csv
// ============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Device code (header-only + two "source" files pulled in as a single TU,
// same strategy as warp_sort_bench.cu).
#include "hull_dandc.cuh"
#include "kdop_hull.cuh"
#include "mesh_volume.cuh"

// heap_init_kernel lives in cuda/mm.cu and KDOP_AXES lives in cuda/kdop_const.cu
// (both linked as separate -dc objects; see Makefile.hull).  Forward-declare
// heap_init_kernel here so we can launch it via the runtime <<<>>> syntax.
extern "C" __global__ void heap_init_kernel(DevicePool* pool);

// Host-side btConvexHullComputer (compiled as separate .cpp objects).
#include "btConvexHullComputer.h"
#include "btVector3.h"

// ============================================================================
// CUDA error check
// ============================================================================
#define CK(expr) do {                                                     \
    cudaError_t _e = (expr);                                              \
    if (_e != cudaSuccess) {                                              \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                     cudaGetErrorString(_e), __FILE__, __LINE__, #expr);  \
        std::exit(1);                                                     \
    }                                                                     \
} while (0)

// ============================================================================
// Point generators
// ============================================================================
enum class Dist { Gaussian, UniformCube };
static const char* dist_name(Dist d) {
    return d == Dist::Gaussian ? "gaussian" : "uniform_cube";
}

static void gen_points(std::vector<float>& out, int n_pts, Dist d,
                       std::mt19937& rng)
{
    out.resize((size_t)n_pts * 3);
    if (d == Dist::Gaussian) {
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : out) v = nd(rng);
    } else {
        std::uniform_real_distribution<float> ud(-0.5f, 0.5f);
        for (auto& v : out) v = ud(rng);
    }
}

// ============================================================================
// CPU hull volume: divergence-theorem fan-triangulation over the faces
// returned by btConvexHullComputer.  Matches tests/bench_cpu_hull.cpp.
// ============================================================================
static double cpu_hull_volume(const btConvexHullComputer& ch) {
    double vol = 0.0;
    int nf = ch.faces.size();
    for (int f = 0; f < nf; ++f) {
        const btConvexHullComputer::Edge* e0 = &ch.edges[ch.faces[f]];
        int a = e0->getSourceVertex();
        int b = e0->getTargetVertex();
        const btConvexHullComputer::Edge* e = e0->getNextEdgeOfFace();
        int c = e->getTargetVertex();
        while (c != a) {
            const auto& va = ch.vertices[a];
            const auto& vb = ch.vertices[b];
            const auto& vc = ch.vertices[c];
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

// ============================================================================
// Device kernels (thin wrappers; 1 warp / 1 block per hull).
// ============================================================================
extern "C" __global__ void bench_hull_dandc_kernel(
    const float* __restrict__ pts,
    const int*   __restrict__ offsets,
    int          n_hulls,
    DeviceHeap*  heap,
    DeviceHeap*  scratch_heap,
    int*         out_errors,
    int*         out_nv,
    int*         out_nt)
{
    int warp_id = blockIdx.x;
    int lane    = threadIdx.x & (WARP_SIZE - 1);
    if (warp_id >= n_hulls) return;

    int start = offsets[warp_id];
    int count = offsets[warp_id + 1] - start;

    __shared__ Mesh s_mesh;
    int err = 0;
    hull_dandc_warp_mesh(pts + (long long)start * 3, count, lane,
                         heap, scratch_heap, &err, &s_mesh);
    if (lane == 0) {
        out_errors[warp_id] = err;
        out_nv[warp_id]     = s_mesh.nv;
        out_nt[warp_id]     = s_mesh.nt;
    }
    __syncwarp();
    // Free output allocation immediately -- we only care about timing, not
    // the hull mesh.  Balances heap alloc/free so repeated runs reuse blocks.
    if (lane == 0 && s_mesh.verts) heap_free(heap, s_mesh.verts);
}

extern "C" __global__ void bench_kdop_hull_kernel(
    const float* __restrict__ pts,
    const int*   __restrict__ offsets,
    int          n_hulls,
    DeviceHeap*  heap,
    DeviceHeap*  scratch_heap,
    int*         out_errors,
    int*         out_nv,
    int*         out_nt,
    float*       out_vol)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= n_hulls) return;

    int start = offsets[bid];
    int count = offsets[bid + 1] - start;

    float vol = 0.f;
    int err = 0;
    Mesh mesh = kdop_hull_block(pts + (long long)start * 3, count,
                                heap, scratch_heap, &vol, &err);
    if (tid == 0) {
        out_errors[bid] = err;
        out_nv[bid]     = mesh.nv;
        out_nt[bid]     = mesh.nt;
        out_vol[bid]    = vol;
    }
    __syncthreads();
    if (tid == 0 && mesh.verts) heap_free(heap, mesh.verts);
}

// ============================================================================
// Device pool / heap setup (mirrors csrc/heap.c but using the CUDA runtime).
// ============================================================================
struct DevPool {
    void*               d_pool_mem   = nullptr;
    unsigned long long* d_pool_off   = nullptr;
    DevicePool*         d_pool       = nullptr;
    size_t              capacity     = 0;
};

static void init_pool(DevPool& p, size_t pool_bytes) {
    CK(cudaMalloc(&p.d_pool_mem, pool_bytes));
    CK(cudaMalloc((void**)&p.d_pool_off, sizeof(unsigned long long)));
    CK(cudaMalloc((void**)&p.d_pool,     sizeof(DevicePool)));
    CK(cudaMemset(p.d_pool, 0, sizeof(DevicePool)));
    p.capacity = pool_bytes;

    DevicePool host_dp{};
    host_dp.base     = (char*)p.d_pool_mem;
    host_dp.offset   = p.d_pool_off;
    host_dp.capacity = (unsigned long long)pool_bytes;
    // Only the leading scalar fields (before the embedded heaps) need upload;
    // the embedded heaps are zeroed above and then filled by heap_init_kernel.
    CK(cudaMemcpy(p.d_pool, &host_dp, offsetof(DevicePool, heap),
                  cudaMemcpyHostToDevice));

    unsigned long long zero = 0;
    CK(cudaMemcpy(p.d_pool_off, &zero, sizeof(zero),
                  cudaMemcpyHostToDevice));

    // heap_init_kernel: 2 * HEAP_NUM_ARENAS blocks (one per arena × 2 heaps).
    heap_init_kernel<<<HEAP_NUM_ARENAS * 2, 32>>>(p.d_pool);
    CK(cudaDeviceSynchronize());
}

static void reset_pool_offset(DevPool& p) {
    unsigned long long zero = 0;
    CK(cudaMemcpy(p.d_pool_off, &zero, sizeof(zero),
                  cudaMemcpyHostToDevice));
    // Re-run arena init so any free-list state from the previous run is wiped.
    heap_init_kernel<<<HEAP_NUM_ARENAS * 2, 32>>>(p.d_pool);
    CK(cudaDeviceSynchronize());
}

static void free_pool(DevPool& p) {
    if (p.d_pool_mem) cudaFree(p.d_pool_mem);
    if (p.d_pool_off) cudaFree(p.d_pool_off);
    if (p.d_pool)     cudaFree(p.d_pool);
    p = {};
}

// Device addresses of the embedded heaps within d_pool.
static inline DeviceHeap* d_heap_ptr(DevPool& p) {
    return (DeviceHeap*)((char*)p.d_pool + offsetof(DevicePool, heap));
}
static inline DeviceHeap* d_scratch_ptr(DevPool& p) {
    return (DeviceHeap*)((char*)p.d_pool + offsetof(DevicePool, scratch));
}

// ============================================================================
// Timing helpers (pristine input restored before each measured run).
// ============================================================================
struct TimingResult {
    double mean_ms;
    double std_ms;
    bool   ok;
};

template<typename SetupFn, typename BodyFn>
TimingResult time_cuda(SetupFn setup, BodyFn body, int warmup, int measure) {
    for (int i = 0; i < warmup; i++) { setup(); body(); }
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    std::vector<double> times(measure);
    for (int i = 0; i < measure; i++) {
        setup();
        CK(cudaEventRecord(e0));
        body();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());     // surface any async kernel error
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        times[i] = ms;
    }
    CK(cudaDeviceSynchronize());
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));

    double m = 0;  for (double t : times) m += t;  m /= measure;
    double v = 0;  for (double t : times) v += (t - m) * (t - m);
    v /= measure;
    return {m, std::sqrt(v), true};
}

template<typename SetupFn, typename BodyFn>
TimingResult time_host(SetupFn setup, BodyFn body, int warmup, int measure) {
    for (int i = 0; i < warmup; i++) { setup(); body(); }
    std::vector<double> times(measure);
    for (int i = 0; i < measure; i++) {
        setup();
        auto t0 = std::chrono::high_resolution_clock::now();
        body();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double m = 0;  for (double t : times) m += t;  m /= measure;
    double v = 0;  for (double t : times) v += (t - m) * (t - m);
    v /= measure;
    return {m, std::sqrt(v), true};
}

// Kernel attributes (regs, static smem) for reporting.
struct KernelInfo { int regs; int smem; };
static KernelInfo kinfo(const void* f) {
    if (!f) return {-1, -1};
    cudaFuncAttributes a{};
    if (cudaFuncGetAttributes(&a, f) != cudaSuccess) return {-1, -1};
    return {a.numRegs, (int)a.sharedSizeBytes};
}

// ============================================================================
// Row printer
// ============================================================================
static void print_row(const char* algo, const char* dist,
                      int batch, int n_pts, const TimingResult& r,
                      KernelInfo ki, double mean_vol)
{
    if (!r.ok) {
        std::printf("%s,%s,%d,%d,skip,skip,%d,%d,%.6g\n",
                    algo, dist, batch, n_pts, ki.regs, ki.smem, mean_vol);
    } else {
        std::printf("%s,%s,%d,%d,%.4f,%.4f,%d,%d,%.6g\n",
                    algo, dist, batch, n_pts,
                    r.mean_ms, r.std_ms, ki.regs, ki.smem, mean_vol);
    }
    std::fflush(stdout);
}

// ============================================================================
// One case: (dist, batch, n_pts).  Runs all three impls.
// ============================================================================
static constexpr int CUDA_WARMUP  = 2;
static constexpr int CUDA_MEASURE = 5;
static constexpr int HOST_WARMUP  = 1;
static constexpr int HOST_MEASURE = 3;

static void run_case(Dist dist, int batch, int n_pts, DevPool& pool,
                     std::mt19937& rng)
{
    const char* dname = dist_name(dist);
    size_t total = (size_t)batch * n_pts;

    // Generate the batch once (shared by CPU + GPU).
    std::vector<float> h_pts(total * 3);
    std::vector<std::vector<float>> per_hull(batch);
    for (int b = 0; b < batch; b++) {
        gen_points(per_hull[b], n_pts, dist, rng);
        std::memcpy(h_pts.data() + (size_t)b * n_pts * 3,
                    per_hull[b].data(),
                    (size_t)n_pts * 3 * sizeof(float));
    }

    // -------------------------------------------------------------------- CPU
    {
        // Pre-convert to std::vector<std::array<double,3>> for each hull.
        using PtArr = std::array<double, 3>;
        std::vector<std::vector<PtArr>> cpu_pts(batch, std::vector<PtArr>(n_pts));
        for (int b = 0; b < batch; b++) {
            const float* src = per_hull[b].data();
            for (int i = 0; i < n_pts; i++) {
                cpu_pts[b][i] = { (double)src[i*3+0],
                                  (double)src[i*3+1],
                                  (double)src[i*3+2] };
            }
        }

        double vol_sum = 0.0;
        auto setup = [] {};
        auto body = [&] {
            vol_sum = 0.0;
            for (int b = 0; b < batch; b++) {
                btConvexHullComputer ch;
                ch.compute(cpu_pts[b], -1.0, -1.0);
                vol_sum += cpu_hull_volume(ch);
            }
        };
        TimingResult r = time_host(setup, body, HOST_WARMUP, HOST_MEASURE);
        print_row("cpu_bt_hull", dname, batch, n_pts, r, {-1, -1},
                  vol_sum / batch);
    }

    // -------------------------------------------------------------------- GPU
    // Offsets (uniform-length batch).
    std::vector<int> h_offs(batch + 1);
    for (int i = 0; i <= batch; i++) h_offs[i] = i * n_pts;

    float* d_pts  = nullptr;
    int*   d_offs = nullptr;
    int*   d_err  = nullptr;
    int*   d_nv   = nullptr;
    int*   d_nt   = nullptr;
    float* d_vol  = nullptr;
    CK(cudaMalloc(&d_pts,  total * 3 * sizeof(float)));
    CK(cudaMalloc(&d_offs, (size_t)(batch + 1) * sizeof(int)));
    CK(cudaMalloc(&d_err,  (size_t)batch * sizeof(int)));
    CK(cudaMalloc(&d_nv,   (size_t)batch * sizeof(int)));
    CK(cudaMalloc(&d_nt,   (size_t)batch * sizeof(int)));
    CK(cudaMalloc(&d_vol,  (size_t)batch * sizeof(float)));
    CK(cudaMemcpy(d_pts,  h_pts.data(),  total * 3 * sizeof(float),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_offs, h_offs.data(), (size_t)(batch + 1) * sizeof(int),
                  cudaMemcpyHostToDevice));

    DeviceHeap* d_h  = d_heap_ptr(pool);
    DeviceHeap* d_sc = d_scratch_ptr(pool);

    // ---- hull_dandc ----
    {
        auto setup = [&] { reset_pool_offset(pool); };
        auto body = [&] {
            bench_hull_dandc_kernel<<<batch, 32>>>(
                d_pts, d_offs, batch, d_h, d_sc, d_err, d_nv, d_nt);
        };
        TimingResult r = time_cuda(setup, body, CUDA_WARMUP, CUDA_MEASURE);

        // Compute mean volume from last run via mesh_volume: easier to just
        // report 0 here (volume comparison is a sanity check, not the bench).
        // We'll use kdop's out_vol for a GPU-side volume reference instead.
        KernelInfo ki = kinfo((const void*)bench_hull_dandc_kernel);
        print_row("hull_dandc", dname, batch, n_pts, r, ki, 0.0);
    }

    // ---- kdop_hull ----
    {
        auto setup = [&] { reset_pool_offset(pool); };
        auto body = [&] {
            bench_kdop_hull_kernel<<<batch, 32>>>(
                d_pts, d_offs, batch, d_h, d_sc, d_err, d_nv, d_nt, d_vol);
        };
        TimingResult r = time_cuda(setup, body, CUDA_WARMUP, CUDA_MEASURE);

        // Pull back volumes for sanity.
        std::vector<float> h_vol(batch);
        CK(cudaMemcpy(h_vol.data(), d_vol, (size_t)batch * sizeof(float),
                      cudaMemcpyDeviceToHost));
        double vol_sum = 0.0;
        int ok = 0;
        for (int i = 0; i < batch; i++) {
            if (std::isfinite(h_vol[i])) { vol_sum += h_vol[i]; ok++; }
        }
        KernelInfo ki = kinfo((const void*)bench_kdop_hull_kernel);
        print_row("kdop_hull", dname, batch, n_pts, r, ki,
                  ok ? vol_sum / ok : 0.0);
    }

    cudaFree(d_pts);
    cudaFree(d_offs);
    cudaFree(d_err);
    cudaFree(d_nv);
    cudaFree(d_nt);
    cudaFree(d_vol);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    int device = 0;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device = std::atoi(argv[++i]);
    }
    CK(cudaSetDevice(device));

    // Raise default stack frame for D&C recursion (matches gpu_hull_dandc).
    CK(cudaDeviceSetLimit(cudaLimitStackSize, 8 * 1024));

    DevPool pool;
    init_pool(pool, (size_t)6 * 1024 * 1024 * 1024);   // 6 GB

    std::printf("algorithm,dist,batch,n_pts,mean_ms,std_ms,regs,smem,mean_vol\n");

    std::mt19937 rng(0xC0ACDu);

    const int batch   = 256;
    const int n_pts[] = { 512, 2048, 16384, 65536 };
    const Dist dists[] = { Dist::Gaussian, Dist::UniformCube };

    for (Dist d : dists) {
        for (int n : n_pts) {
            run_case(d, batch, n, pool, rng);
        }
    }

    free_pool(pool);
    return 0;
}
