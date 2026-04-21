// ============================================================================
// malloc_test_bench.cu -- GPU port of glibc/mstress-style malloc stress test.
//
// Block-as-CPU-thread model: each block emulates one CPU thread. Only thread 0
// of each block issues heap_alloc / heap_free; other lanes idle (allocator API
// requires thread-0-only calls -- see allocator.cuh:26).
//
// Workload (mstress/glibc bench-malloc-thread style):
//   - Each block owns `slots` pointer slots in global memory (initially null).
//   - Per iteration, thread 0:
//       1. picks a random slot,
//       2. frees whatever ptr is there (if non-null),
//       3. allocates a new log-uniform random size and stores the ptr.
//   - After `iters` iterations it frees all remaining slots.
//
// One alloc-free pair counts as one op; total_ops = grid_blocks * iters (once
// the pipeline is primed; the first `slots-1` iters of each block skip the free
// leg, which is a small correction we ignore in the reported rate).
//
// Sweeps the grid size (block count) so arena contention becomes visible:
// there are HEAP_NUM_ARENAS=64 arenas, block i maps to arena i%64, and each
// arena has a spin-lock (allocator.cuh:121). grid=1 => no contention,
// grid=1024 => ~16 blocks/arena serialised.
//
// Build (from bench/):
//   make -f Makefile.malloc                # ARCH=89 default (RTX 4090)
//   make -f Makefile.malloc verbose        # with ptxas -v
// Run:
//   ./malloc_test_bench                    # defaults (see below)
//   ./malloc_test_bench --size-min 65536 --size-max 8388608 --slots 8 \
//                       --pool-gb 16       # wide-range stress
// ============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "allocator.cuh"

extern "C" __global__ void heap_init_kernel(DevicePool* pool);

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
// Device-side RNG + log-uniform size sampler
// ============================================================================
__device__ __forceinline__ uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

__device__ __forceinline__ float u01(uint32_t r) {
    // top 24 bits -> [0, 1)
    return (r >> 8) * (1.0f / 16777216.0f);
}

// ============================================================================
// Stress kernel: thread 0 per block runs `iters` alloc/free iterations.
// Other lanes return immediately.
// ============================================================================
extern "C" __global__ void mstress_kernel(
    DeviceHeap*  heap,
    void**       slot_arrays,    // grid * slots, zero-initialised
    int          slots,
    int          iters,
    uint32_t     size_min,
    uint32_t     size_max,
    float        ln_min,
    float        ln_delta,
    int*         out_errors)     // one int per block
{
    if (threadIdx.x != 0) return;

    int bid = blockIdx.x;
    void** my_slots = slot_arrays + (long long)bid * slots;

    // Seed with bid + golden ratio constant; avoids degenerate xorshift seed 0.
    uint32_t rng = (uint32_t)(bid + 1) * 2654435761u;

    for (int i = 0; i < iters; ++i) {
        int slot = (int)(xorshift32(rng) % (uint32_t)slots);
        float u  = u01(xorshift32(rng));
        uint32_t sz = (uint32_t)__expf(ln_min + u * ln_delta);
        if (sz < size_min) sz = size_min;
        if (sz > size_max) sz = size_max;

        void* old = my_slots[slot];
        if (old) {
            int r = heap_free(heap, old);
            if (r != HEAP_OK) { out_errors[bid] = r; return; }
        }
        void* p = nullptr;
        int r = heap_alloc(heap, sz, &p);
        if (r != HEAP_OK) { out_errors[bid] = r; return; }
        my_slots[slot] = p;
    }

    // Drain remaining live slots so heap is balanced before the next run.
    for (int i = 0; i < slots; ++i) {
        void* p = my_slots[i];
        if (p) {
            heap_free(heap, p);
            my_slots[i] = nullptr;
        }
    }
}

// ============================================================================
// CUDA __device__ malloc/free variant (same shape, for head-to-head comparison)
// ============================================================================
#define CUDA_MALLOC_ERR (1 << 30)  // bit flag for device-malloc failure

extern "C" __global__ void cuda_malloc_kernel(
    void**       slot_arrays,
    int          slots,
    int          iters,
    uint32_t     size_min,
    uint32_t     size_max,
    float        ln_min,
    float        ln_delta,
    int*         out_errors)
{
    if (threadIdx.x != 0) return;

    int bid = blockIdx.x;
    void** my_slots = slot_arrays + (long long)bid * slots;
    uint32_t rng = (uint32_t)(bid + 1) * 2654435761u;

    for (int i = 0; i < iters; ++i) {
        int slot = (int)(xorshift32(rng) % (uint32_t)slots);
        float u  = u01(xorshift32(rng));
        uint32_t sz = (uint32_t)__expf(ln_min + u * ln_delta);
        if (sz < size_min) sz = size_min;
        if (sz > size_max) sz = size_max;

        void* old = my_slots[slot];
        if (old) free(old);
        void* p = malloc((size_t)sz);
        if (!p) { out_errors[bid] = CUDA_MALLOC_ERR; return; }
        my_slots[slot] = p;
    }
    for (int i = 0; i < slots; ++i) {
        if (my_slots[i]) { free(my_slots[i]); my_slots[i] = nullptr; }
    }
}

// ============================================================================
// Device pool / heap setup (mirror of hull_bench.cu)
// ============================================================================
struct DevPool {
    void*               d_pool_mem = nullptr;
    unsigned long long* d_pool_off = nullptr;
    DevicePool*         d_pool     = nullptr;
    size_t              capacity   = 0;
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
    CK(cudaMemcpy(p.d_pool, &host_dp, offsetof(DevicePool, heap),
                  cudaMemcpyHostToDevice));

    unsigned long long zero = 0;
    CK(cudaMemcpy(p.d_pool_off, &zero, sizeof(zero), cudaMemcpyHostToDevice));
    heap_init_kernel<<<HEAP_NUM_ARENAS * 2, 32>>>(p.d_pool);
    CK(cudaDeviceSynchronize());
}

static void reset_pool(DevPool& p) {
    unsigned long long zero = 0;
    CK(cudaMemcpy(p.d_pool_off, &zero, sizeof(zero), cudaMemcpyHostToDevice));
    heap_init_kernel<<<HEAP_NUM_ARENAS * 2, 32>>>(p.d_pool);
    CK(cudaDeviceSynchronize());
}

static void free_pool(DevPool& p) {
    if (p.d_pool_mem) cudaFree(p.d_pool_mem);
    if (p.d_pool_off) cudaFree(p.d_pool_off);
    if (p.d_pool)     cudaFree(p.d_pool);
    p = {};
}

static inline DeviceHeap* d_heap_ptr(DevPool& p) {
    return (DeviceHeap*)((char*)p.d_pool + offsetof(DevicePool, heap));
}

// ============================================================================
// Error-code name (best-effort; unknown codes -> numeric)
// ============================================================================
static const char* kerr_name(int code) {
    switch (code) {
        case HEAP_OK: return "OK";
        case KERR_HEAP_OOM: return "HEAP_OOM";
        case KERR_HEAP_CORRUPT: return "HEAP_CORRUPT";
        case KERR_HEAP_CORRUPT_PREV: return "HEAP_CORRUPT_PREV";
        case CUDA_MALLOC_ERR: return "CUDA_MALLOC_NULL";
        default: return "?";
    }
}

// ============================================================================
// CLI parsing
// ============================================================================
struct Opts {
    std::vector<int> grids   = {1, 4, 16, 64, 256, 1024};
    uint32_t size_min        = 4 * 1024;       // 4 KB
    uint32_t size_max        = 256 * 1024;     // 256 KB
    int      slots           = 16;
    int      iters           = 1024;
    double   pool_gb         = 4.0;
    int      warmup          = 2;
    int      measure         = 5;
    std::string impl         = "both";         // "coacd" | "cuda" | "both"
};

static void parse_grids(const char* s, std::vector<int>& out) {
    out.clear();
    std::string tok;
    for (const char* p = s; ; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
            tok.clear();
            if (*p == '\0') break;
        } else {
            tok.push_back(*p);
        }
    }
}

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s [options]\n"
        "  --grids CSV           comma-separated block counts (default 1,4,16,64,256,1024)\n"
        "  --size-min BYTES      minimum allocation size (default 4096)\n"
        "  --size-max BYTES      maximum allocation size (default 262144)\n"
        "  --slots N             live slots per block (default 16)\n"
        "  --iters N             iterations per block (default 1024)\n"
        "  --pool-gb F           pool capacity in GiB (default 4)\n"
        "  --warmup N            warmup runs (default 2)\n"
        "  --measure N           measurement runs (default 5)\n"
        "  --impl {coacd|cuda|both}  which allocator(s) to bench (default both)\n", argv0);
}

static Opts parse(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name){
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if      (a == "--grids")     parse_grids(need("--grids"), o.grids);
        else if (a == "--size-min")  o.size_min = (uint32_t)std::strtoull(need("--size-min"), nullptr, 10);
        else if (a == "--size-max")  o.size_max = (uint32_t)std::strtoull(need("--size-max"), nullptr, 10);
        else if (a == "--slots")     o.slots    = std::atoi(need("--slots"));
        else if (a == "--iters")     o.iters    = std::atoi(need("--iters"));
        else if (a == "--pool-gb")   o.pool_gb  = std::atof(need("--pool-gb"));
        else if (a == "--warmup")    o.warmup   = std::atoi(need("--warmup"));
        else if (a == "--measure")   o.measure  = std::atoi(need("--measure"));
        else if (a == "--impl")      o.impl     = need("--impl");
        else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); std::exit(2); }
    }
    if (o.size_min == 0 || o.size_max < o.size_min) {
        std::fprintf(stderr, "bad size range\n"); std::exit(2);
    }
    return o;
}

// ============================================================================
// Run one config
// ============================================================================
struct Stat { double mean_ms; double std_ms; int err_code; };

static Stat run_one(const Opts& o, int grid, DevPool& pool, const char* impl) {
    int max_grid = *std::max_element(o.grids.begin(), o.grids.end());
    size_t slot_bytes = (size_t)max_grid * o.slots * sizeof(void*);
    static void** d_slots = nullptr;
    static int*   d_errs  = nullptr;
    static size_t cur_slot_bytes = 0;
    static size_t cur_err_bytes  = 0;
    if (slot_bytes > cur_slot_bytes) {
        if (d_slots) cudaFree(d_slots);
        CK(cudaMalloc((void**)&d_slots, slot_bytes));
        cur_slot_bytes = slot_bytes;
    }
    size_t err_bytes = (size_t)max_grid * sizeof(int);
    if (err_bytes > cur_err_bytes) {
        if (d_errs) cudaFree(d_errs);
        CK(cudaMalloc((void**)&d_errs, err_bytes));
        cur_err_bytes = err_bytes;
    }

    float ln_min   = std::log((float)o.size_min);
    float ln_delta = std::log((float)o.size_max) - ln_min;

    bool is_coacd = (std::strcmp(impl, "coacd") == 0);
    auto setup = [&]() {
        if (is_coacd) reset_pool(pool);
        // For CUDA device malloc, the runtime heap is owned by the driver;
        // we cannot reset it between runs. See caveat in main().
        CK(cudaMemset(d_slots, 0, (size_t)grid * o.slots * sizeof(void*)));
        CK(cudaMemset(d_errs,  0, (size_t)grid * sizeof(int)));
    };
    auto body = [&]() {
        if (is_coacd) {
            mstress_kernel<<<grid, 32>>>(
                d_heap_ptr(pool), d_slots, o.slots, o.iters,
                o.size_min, o.size_max, ln_min, ln_delta, d_errs);
        } else {
            cuda_malloc_kernel<<<grid, 32>>>(
                d_slots, o.slots, o.iters,
                o.size_min, o.size_max, ln_min, ln_delta, d_errs);
        }
    };

    for (int i = 0; i < o.warmup; ++i) { setup(); body(); }
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::vector<double> times(o.measure);
    for (int i = 0; i < o.measure; ++i) {
        setup();
        CK(cudaEventRecord(e0));
        body();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        times[i] = ms;
    }
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));

    // Check errors from the last run.
    std::vector<int> h_errs((size_t)grid, 0);
    CK(cudaMemcpy(h_errs.data(), d_errs, (size_t)grid * sizeof(int),
                  cudaMemcpyDeviceToHost));
    int first_err = 0;
    for (int e : h_errs) { if (e) { first_err = e; break; } }

    double m = 0;  for (double t : times) m += t;  m /= o.measure;
    double v = 0;  for (double t : times) v += (t - m) * (t - m);
    v /= o.measure;
    return {m, std::sqrt(v), first_err};
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    Opts o = parse(argc, argv);

    size_t pool_bytes = (size_t)(o.pool_gb * (1ULL << 30));
    std::fprintf(stderr,
        "pool=%.2f GiB  size=[%u, %u] B  slots=%d  iters=%d  grids=",
        o.pool_gb, o.size_min, o.size_max, o.slots, o.iters);
    for (size_t i = 0; i < o.grids.size(); ++i)
        std::fprintf(stderr, "%s%d", i ? "," : "", o.grids[i]);
    std::fprintf(stderr, "\n");

    std::vector<std::string> impls;
    if      (o.impl == "both")  { impls = {"coacd", "cuda"}; }
    else if (o.impl == "coacd") { impls = {"coacd"}; }
    else if (o.impl == "cuda")  { impls = {"cuda"}; }
    else { std::fprintf(stderr, "bad --impl: %s\n", o.impl.c_str()); return 2; }

    std::printf("impl,grid,size_min,size_max,slots,iters,mean_ms,std_ms,"
                "ops_per_sec,per_block_ops_per_sec,err\n");

    // Run each impl in isolation: allocate its heap, sweep grids, tear it
    // down before moving on. Otherwise the two heaps (coacd pool + CUDA
    // device-malloc heap) compete for VRAM at full pool-gb each.
    for (const auto& impl : impls) {
        DevPool pool{};
        bool is_coacd = (impl == "coacd");
        if (is_coacd) {
            init_pool(pool, pool_bytes);
        } else {
            cudaError_t e = cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                                               pool_bytes);
            if (e != cudaSuccess) {
                std::fprintf(stderr,
                    "warning: cudaDeviceSetLimit(MallocHeapSize, %zu) -> %s\n",
                    pool_bytes, cudaGetErrorString(e));
            }
        }

        for (int grid : o.grids) {
            Stat s = run_one(o, grid, pool, impl.c_str());
            double total_ops = (double)grid * (double)o.iters;
            double ops_per_sec = total_ops / (s.mean_ms * 1e-3);
            double per_block   = (double)o.iters / (s.mean_ms * 1e-3);
            std::printf("%s,%d,%u,%u,%d,%d,%.4f,%.4f,%.3e,%.3e,%s\n",
                        impl.c_str(), grid, o.size_min, o.size_max, o.slots, o.iters,
                        s.mean_ms, s.std_ms, ops_per_sec, per_block,
                        kerr_name(s.err_code));
            std::fflush(stdout);
        }

        if (is_coacd) free_pool(pool);
    }
    return 0;
}
