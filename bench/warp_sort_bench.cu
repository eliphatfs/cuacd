// ============================================================================
// warp_sort_bench.cu -- standalone perf bench for warp-cooperative sort.
//
// Compares:
//   host:   std::sort
//   device: our warp_sort (warp_sort.cuh)
//   device: cub::BlockMergeSort
//   device: cub::BlockRadixSort  (skipped for int4: non-arithmetic key)
//
// Dtypes: float, int, int4
// Batch:  1, 100, 10000
// Seq:    100, 1000, 10000, 100000  (skip batch=10000 x seq=100000)
// Block:  32, 128
//
// For block=128 our_sort_b128:
//   phase 1  warp 0 picks pivot + 3-way partition on full sequence
//   phase 2  warp 0 partitions the left sub-range, warp 1 the right sub-range
//   phase 3  warps 0..3 each warp_sort_inner their segment in parallel
//
// Scratch is in global memory for every cuda impl (CUB's BlockXSort still
// uses its required __shared__ temp_storage, nothing more).
//
// Build (from repo root):
//   nvcc -std=c++17 -O3 -arch=sm_89 -I cuda bench/warp_sort_bench.cu \
//        -o bench/warp_sort_bench
// Run:
//   ./bench/warp_sort_bench > bench/warp_sort_results.csv
// ============================================================================

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

// Bring in the device-side sort template. Transitively pulls allocator.cuh /
// warp_common.cuh but only for inline __device__ helpers; no global kernels
// or host functions from the project are linked.
#include "warp_sort.cuh"

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
// Comparators -- device + host views
// ============================================================================
struct FloatCmp {
    static __device__ inline int cmp(float a, float b) {
        return (a < b) ? -1 : (a > b) ? 1 : 0;
    }
    static __device__ inline float sentinel() {
        return std::numeric_limits<float>::infinity();
    }
    __device__ __host__ inline bool operator()(float a, float b) const {
        return a < b;
    }
};

struct IntCmp {
    static __device__ inline int cmp(int a, int b) {
        return (a < b) ? -1 : (a > b) ? 1 : 0;
    }
    static __device__ inline int sentinel() {
        return std::numeric_limits<int>::max();
    }
    __device__ __host__ inline bool operator()(int a, int b) const {
        return a < b;
    }
};

struct Int4Cmp {
    static __device__ inline int cmp(int4 a, int4 b) {
        if (a.x != b.x) return (a.x < b.x) ? -1 : 1;
        if (a.y != b.y) return (a.y < b.y) ? -1 : 1;
        if (a.z != b.z) return (a.z < b.z) ? -1 : 1;
        if (a.w != b.w) return (a.w < b.w) ? -1 : 1;
        return 0;
    }
    static __device__ inline int4 sentinel() {
        int4 s; s.x = s.y = s.z = s.w = std::numeric_limits<int>::max();
        return s;
    }
    __device__ __host__ inline bool operator()(int4 a, int4 b) const {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        if (a.z != b.z) return a.z < b.z;
        return a.w < b.w;
    }
};

template<typename T> struct CmpFor;
template<> struct CmpFor<float> { using type = FloatCmp; };
template<> struct CmpFor<int>   { using type = IntCmp;   };
template<> struct CmpFor<int4>  { using type = Int4Cmp;  };

template<typename T> const char* dtype_name();
template<> const char* dtype_name<float>() { return "float"; }
template<> const char* dtype_name<int>()   { return "int"; }
template<> const char* dtype_name<int4>()  { return "int4"; }

// ============================================================================
// Host data generation (normal distribution)
// ============================================================================
template<typename T>
void gen_normal(std::vector<T>& out, std::mt19937& rng);

template<>
void gen_normal<float>(std::vector<float>& out, std::mt19937& rng) {
    std::normal_distribution<float> d(0.f, 1.f);
    for (auto& v : out) v = d(rng);
}

template<>
void gen_normal<int>(std::vector<int>& out, std::mt19937& rng) {
    std::normal_distribution<float> d(0.f, 1e6f);
    for (auto& v : out) v = (int)std::llround(d(rng));
}

template<>
void gen_normal<int4>(std::vector<int4>& out, std::mt19937& rng) {
    std::normal_distribution<float> d(0.f, 1e6f);
    for (auto& v : out) {
        v.x = (int)std::llround(d(rng));
        v.y = (int)std::llround(d(rng));
        v.z = (int)std::llround(d(rng));
        v.w = (int)std::llround(d(rng));
    }
}

// ============================================================================
// Our warp_sort kernels (block=32 and block=128 variants)
// ============================================================================
template<typename T, typename Cmp>
__global__ void our_sort_b32_kernel(T* data, char* scratch_all,
                                     int n, size_t scratch_stride)
{
    int bid  = blockIdx.x;
    int lane = threadIdx.x;
    T*   my_data    = data + (size_t)bid * n;
    char* my_scratch = scratch_all + (size_t)bid * scratch_stride;
    warp_sort_t<T, Cmp>(my_data, my_scratch, n, lane);
}

// Scratch layout for b128:
//   [tmp: n*sizeof(T) bytes] [4 warps * 2 stacks * WS_MAX_STACK * sizeof(int)]
template<typename T, typename Cmp>
__global__ void our_sort_b128_kernel(T* data, char* scratch_all,
                                      int n, size_t scratch_stride)
{
    int bid     = blockIdx.x;
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;

    T*   my_data    = data + (size_t)bid * n;
    char* my_scratch = scratch_all + (size_t)bid * scratch_stride;

    T*   tmp        = (T*)my_scratch;
    int* stack_base = (int*)(my_scratch + (size_t)n * sizeof(T));

    __shared__ int s[6];  // mid_lo1, mid_hi1, mlL, mhL, mlR, mhR

    // Phase 1: warp 0 partitions [0, n)
    if (warp_id == 0) {
        if (n > 1) {
            T piv = warp_pick_pivot<T, Cmp>(my_data, 0, n, lane);
            int mlo, mhi;
            warp_partition<T, Cmp>(my_data, tmp, 0, n, piv, &mlo, &mhi, lane);
            if (lane == 0) { s[0] = mlo; s[1] = mhi; }
        } else if (lane == 0) {
            s[0] = 0; s[1] = n;
        }
    }
    __syncthreads();
    int mlo1 = s[0], mhi1 = s[1];

    // Phase 2: warp 0 partitions [0, mlo1), warp 1 partitions [mhi1, n)
    if (warp_id == 0) {
        int lo = 0, hi = mlo1;
        if (hi - lo > 1) {
            T piv = warp_pick_pivot<T, Cmp>(my_data, lo, hi - lo, lane);
            int mlo, mhi;
            warp_partition<T, Cmp>(my_data, tmp, lo, hi, piv, &mlo, &mhi, lane);
            if (lane == 0) { s[2] = mlo; s[3] = mhi; }
        } else if (lane == 0) {
            s[2] = lo; s[3] = hi;
        }
    } else if (warp_id == 1) {
        int lo = mhi1, hi = n;
        if (hi - lo > 1) {
            T piv = warp_pick_pivot<T, Cmp>(my_data, lo, hi - lo, lane);
            int mlo, mhi;
            warp_partition<T, Cmp>(my_data, tmp, lo, hi, piv, &mlo, &mhi, lane);
            if (lane == 0) { s[4] = mlo; s[5] = mhi; }
        } else if (lane == 0) {
            s[4] = lo; s[5] = hi;
        }
    }
    __syncthreads();

    // 4 segments now:
    //   [0, s[2])      [s[3], mlo1)      [mhi1, s[4])      [s[5], n)
    int seg_lo, seg_hi;
    switch (warp_id) {
        case 0: seg_lo = 0;     seg_hi = s[2];  break;
        case 1: seg_lo = s[3];  seg_hi = mlo1;  break;
        case 2: seg_lo = mhi1;  seg_hi = s[4];  break;
        case 3: seg_lo = s[5];  seg_hi = n;     break;
        default: seg_lo = 0;    seg_hi = 0;     break;
    }
    if (warp_id < 4 && seg_hi - seg_lo > 1) {
        int* stk_lo = stack_base + warp_id * 2 * WS_MAX_STACK;
        int* stk_hi = stk_lo + WS_MAX_STACK;
        warp_sort_inner<T, Cmp>(my_data, tmp, stk_lo, stk_hi,
                                seg_lo, seg_hi, lane);
    }
}

template<typename T>
size_t our_scratch_bytes(int n, int block) {
    size_t data_sz = (size_t)n * sizeof(T);
    int warps = block / 32;
    size_t stack_sz = (size_t)warps * 2 * WS_MAX_STACK * sizeof(int);
    return data_sz + stack_sz;
}

// ============================================================================
// CUB block sort kernels
// ============================================================================
template<typename T, int BS, int IPT, typename Cmp>
__global__ void cub_merge_kernel(T* data, int n) {
    using BMS = cub::BlockMergeSort<T, BS, IPT>;
    __shared__ typename BMS::TempStorage ts;
    int bid = blockIdx.x;
    T*  my  = data + (size_t)bid * n;

    T items[IPT];
    Cmp cmp;
    T sent = Cmp::sentinel();
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;  // blocked arrangement
        items[i] = (idx < n) ? my[idx] : sent;
    }
    BMS(ts).Sort(items, cmp);
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        if (idx < n) my[idx] = items[i];
    }
}

template<typename T, int BS, int IPT>
__global__ void cub_radix_kernel(T* data, int n) {
    using BRS = cub::BlockRadixSort<T, BS, IPT>;
    __shared__ typename BRS::TempStorage ts;
    int bid = blockIdx.x;
    T*  my  = data + (size_t)bid * n;

    T items[IPT];
    T sent = std::numeric_limits<T>::max();
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        items[i] = (idx < n) ? my[idx] : sent;
    }
    BRS(ts).Sort(items);
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        if (idx < n) my[idx] = items[i];
    }
}

// CUB is register-expensive: IPT * sizeof(T) bytes sit in each thread's
// register file.  We only instantiate (block, seq, IPT) combos that stay
// within a sensible budget.  Everything else is reported as "skip".
//
// IPT budget ~ 32 scalars / thread for struct-sized types, ~ 80 for 4-byte.
template<typename T, typename Cmp>
bool launch_cub_merge(int bs, int batch, int seq, T* d_data) {
    #define C(BS_, N_, IPT_) \
        if (bs == BS_ && seq == N_) { \
            cub_merge_kernel<T, BS_, IPT_, Cmp><<<batch, BS_>>>(d_data, seq); \
            return true; \
        }
    C(32,  100, 4);
    C(32,  1000, 32);
    C(128, 100, 1);
    C(128, 1000, 8);
    if constexpr (sizeof(T) <= 4) {
        C(128, 10000, 79);
    }
    #undef C
    return false;
}

template<typename T, typename Cmp>
bool supports_cub_merge(int bs, int seq) {
    #define C(BS_, N_) if (bs == BS_ && seq == N_) return true;
    C(32,  100);  C(32,  1000);
    C(128, 100);  C(128, 1000);
    if constexpr (sizeof(T) <= 4) { C(128, 10000); }
    #undef C
    return false;
}

template<typename T>
bool launch_cub_radix(int bs, int batch, int seq, T* d_data) {
    #define C(BS_, N_, IPT_) \
        if (bs == BS_ && seq == N_) { \
            cub_radix_kernel<T, BS_, IPT_><<<batch, BS_>>>(d_data, seq); \
            return true; \
        }
    C(32,  100, 4);
    C(32,  1000, 32);
    C(128, 100, 1);
    C(128, 1000, 8);
    C(128, 10000, 79);
    #undef C
    return false;
}

template<typename T>
bool supports_cub_radix(int bs, int seq) {
    #define C(BS_, N_) if (bs == BS_ && seq == N_) return true;
    C(32, 100);  C(32, 1000);
    C(128, 100); C(128, 1000); C(128, 10000);
    #undef C
    return false;
}
// int4 radix sort: CUB BlockRadixSort only supports arithmetic keys.
template<> bool launch_cub_radix<int4>(int, int, int, int4*) { return false; }
template<> bool supports_cub_radix<int4>(int, int) { return false; }

// ============================================================================
// Timing helpers -- pristine data is restored before every measured run.
// ============================================================================
struct TimingResult {
    double mean_ms;
    double std_ms;
    bool   ok;
};

template<typename SetupFn, typename BodyFn>
TimingResult time_cuda(SetupFn setup, BodyFn body, int warmup, int measure) {
    for (int i = 0; i < warmup; i++) {
        setup();
        body();
    }
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    std::vector<double> times(measure);
    for (int i = 0; i < measure; i++) {
        setup();                       // async memcpy on default stream
        CK(cudaEventRecord(e0));
        body();                        // kernel launch on default stream
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        times[i] = ms;
    }
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));

    double m = 0;
    for (double t : times) m += t;
    m /= measure;
    double v = 0;
    for (double t : times) v += (t - m) * (t - m);
    v /= measure;
    return {m, std::sqrt(v), true};
}

template<typename SetupFn, typename BodyFn>
TimingResult time_host(SetupFn setup, BodyFn body, int warmup, int measure) {
    for (int i = 0; i < warmup; i++) {
        setup();
        body();
    }
    std::vector<double> times(measure);
    for (int i = 0; i < measure; i++) {
        setup();
        auto t0 = std::chrono::high_resolution_clock::now();
        body();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double m = 0;
    for (double t : times) m += t;
    m /= measure;
    double v = 0;
    for (double t : times) v += (t - m) * (t - m);
    v /= measure;
    return {m, std::sqrt(v), true};
}

// ============================================================================
// Case runner
// ============================================================================
static constexpr int CUDA_WARMUP  = 3;
static constexpr int CUDA_MEASURE = 10;
static constexpr int HOST_WARMUP  = 1;
static constexpr int HOST_MEASURE = 3;

static void print_row(const char* algo, const char* dtype, const char* block,
                      int batch, int seq, const TimingResult& r)
{
    if (!r.ok) {
        std::printf("%s,%s,%s,%d,%d,skip,skip\n",
                    algo, dtype, block, batch, seq);
    } else {
        std::printf("%s,%s,%s,%d,%d,%.4f,%.4f\n",
                    algo, dtype, block, batch, seq, r.mean_ms, r.std_ms);
    }
    std::fflush(stdout);
}

template<typename T>
void run_case(int batch, int seq, std::mt19937& rng,
              bool run_host, bool run_cuda)
{
    using Cmp = typename CmpFor<T>::type;
    const char* dt = dtype_name<T>();
    size_t total = (size_t)batch * seq;

    // ---- host pristine data ----
    std::vector<T> h_pristine(total);
    gen_normal<T>(h_pristine, rng);

    // ---- host sort timing ----
    if (run_host) {
        std::vector<T> h_work(total);
        auto setup = [&] { std::memcpy(h_work.data(), h_pristine.data(),
                                       total * sizeof(T)); };
        auto body  = [&] {
            Cmp cmp;
            for (int b = 0; b < batch; b++) {
                std::sort(h_work.data() + (size_t)b * seq,
                          h_work.data() + (size_t)(b + 1) * seq, cmp);
            }
        };
        TimingResult r = time_host(setup, body, HOST_WARMUP, HOST_MEASURE);
        print_row("std_sort", dt, "-", batch, seq, r);
    }

    if (!run_cuda) return;

    // ---- device pristine + working buffers ----
    T* d_pristine = nullptr;
    T* d_work     = nullptr;
    CK(cudaMalloc(&d_pristine, total * sizeof(T)));
    CK(cudaMalloc(&d_work,     total * sizeof(T)));
    CK(cudaMemcpy(d_pristine, h_pristine.data(), total * sizeof(T),
                  cudaMemcpyHostToDevice));

    auto reset = [&] {
        CK(cudaMemcpyAsync(d_work, d_pristine, total * sizeof(T),
                           cudaMemcpyDeviceToDevice));
    };

    for (int bs : {32, 128}) {
        char block_str[16];
        std::snprintf(block_str, sizeof(block_str), "%d", bs);

        // ---- our warp_sort ----
        {
            size_t stride = our_scratch_bytes<T>(seq, bs);
            // 16-byte align just in case.
            stride = (stride + 15) & ~size_t(15);
            char* d_scratch = nullptr;
            CK(cudaMalloc(&d_scratch, stride * (size_t)batch));

            auto body = [&] {
                if (bs == 32) {
                    our_sort_b32_kernel<T, Cmp><<<batch, 32>>>(
                        d_work, d_scratch, seq, stride);
                } else {
                    our_sort_b128_kernel<T, Cmp><<<batch, 128>>>(
                        d_work, d_scratch, seq, stride);
                }
            };
            TimingResult r = time_cuda(reset, body, CUDA_WARMUP, CUDA_MEASURE);
            print_row("warp_sort", dt, block_str, batch, seq, r);

            CK(cudaFree(d_scratch));
        }

        // ---- cub BlockMergeSort ----
        if (supports_cub_merge<T, Cmp>(bs, seq)) {
            auto body = [&] { launch_cub_merge<T, Cmp>(bs, batch, seq, d_work); };
            TimingResult r = time_cuda(reset, body, CUDA_WARMUP, CUDA_MEASURE);
            print_row("cub_merge", dt, block_str, batch, seq, r);
        } else {
            print_row("cub_merge", dt, block_str, batch, seq, {0, 0, false});
        }

        // ---- cub BlockRadixSort ----
        if (supports_cub_radix<T>(bs, seq)) {
            auto body = [&] { launch_cub_radix<T>(bs, batch, seq, d_work); };
            TimingResult r = time_cuda(reset, body, CUDA_WARMUP, CUDA_MEASURE);
            print_row("cub_radix", dt, block_str, batch, seq, r);
        } else {
            print_row("cub_radix", dt, block_str, batch, seq, {0, 0, false});
        }
    }

    CK(cudaFree(d_work));
    CK(cudaFree(d_pristine));
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    int device = 0;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        }
    }
    CK(cudaSetDevice(device));

    std::printf("algorithm,dtype,block,batch,seq,mean_ms,std_ms\n");

    std::mt19937 rng(0xC0ACDu);

    const int batches[] = {1, 100, 10000};
    const int seqs[]    = {100, 1000, 10000, 100000};

    auto run_all_dtypes = [&](int batch, int seq) {
        bool run_host = true;
        bool run_cuda = true;
        run_case<float>(batch, seq, rng, run_host, run_cuda);
        run_case<int>  (batch, seq, rng, run_host, run_cuda);
        run_case<int4> (batch, seq, rng, run_host, run_cuda);
    };

    for (int batch : batches) {
        for (int seq : seqs) {
            if (batch == 10000 && seq == 100000) continue;
            run_all_dtypes(batch, seq);
        }
    }

    return 0;
}
