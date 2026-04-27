// warp_sort.cuh — Generic warp-cooperative quicksort template.
//
// Template API:
//   warp_sort_t<T, Cmp>(data, scratch, n, lane) -> int (0=ok, 1=stack overflow)
//
//   T:   element type, sizeof(T) must be a multiple of 4 bytes
//   Cmp: comparator struct with two static __device__ methods:
//        int cmp(T a, T b)  — returns -1, 0, or 1
//        T   sentinel()     — value that sorts after all valid elements
//
// All 32 lanes must call. Uses a global-memory workspace for partitioning:
//   scratch bytes = n * sizeof(T) + WS_MAX_STACK * 2 * sizeof(int)
// Segments <= 32 elements use a bitonic network; larger segments use
// quicksort partitioning with a median-of-32-samples pivot and three-way
// partition (< pivot left, > pivot right, == pivot middle).
//
// Legacy API preserved:
//   warp_sort_bp32(points, scratch, n, lane)           — sorts BtPoint32
//   ws_cmp / ws_min / ws_max / ws_bitonic32            — BtPoint32 helpers
//
// Defines BtPoint32 (canonical definition used by hull_dandc.cuh too).
// Requires: warp_common.cuh (WARP_SIZE, WARP_MASK)
#pragma once
#include "warp_common.cuh"

// ============================================================================
// Generic warp shuffle for any POD type (sizeof(T) % 4 == 0)
// ============================================================================

template<typename T>
__device__ inline T ws_shfl_xor_t(T val, int xor_mask) {
    union { T v; int i[sizeof(T) / 4]; } u, r;
    u.v = val;
    #pragma unroll
    for (int k = 0; k < (int)(sizeof(T) / 4); k++)
        r.i[k] = __shfl_xor_sync(WARP_MASK, u.i[k], xor_mask);
    return r.v;
}

template<typename T>
__device__ inline T ws_shfl_t(T val, int src_lane) {
    union { T v; int i[sizeof(T) / 4]; } u, r;
    u.v = val;
    #pragma unroll
    for (int k = 0; k < (int)(sizeof(T) / 4); k++)
        r.i[k] = __shfl_sync(WARP_MASK, u.i[k], src_lane);
    return r.v;
}

// ============================================================================
// Generic bitonic sort for <= 32 elements (all lanes participate)
// ============================================================================

template<typename T, typename Cmp>
__device__ inline T warp_bitonic32_t(T val, int n, int lane) {
    if (lane >= n) val = Cmp::sentinel();

    for (int k = 1; k <= 16; k <<= 1) {
        for (int j = k; j >= 1; j >>= 1) {
            T other = ws_shfl_xor_t(val, j);
            bool asc = ((lane & (k << 1)) == 0);
            bool less = (Cmp::cmp(val, other) < 0);
            // lane & j == 0: keep min if ascending, max if descending
            // lane & j != 0: keep max if ascending, min if descending
            if ((lane & j) == 0)
                val = (asc == less) ? val : other;
            else
                val = (asc != less) ? val : other;
        }
    }
    return val;
}

// ============================================================================
// Generic warp quicksort — helpers and main entry point
// ============================================================================

#define WS_MAX_STACK 2048

// Pick pivot for data[seg_lo .. seg_lo+seg_len): sample 32 elements uniformly,
// bitonic sort them, return the median (sample[15]).
// All 32 lanes must call with identical arguments.
template<typename T, typename Cmp>
__device__ inline T warp_pick_pivot(T* data, int seg_lo, int seg_len, int lane) {
    int base_step = seg_len / 32;
    int remainder = seg_len - base_step * 32;
    int idx;
    if (lane < remainder)
        idx = seg_lo + lane * (base_step + 1);
    else
        idx = seg_lo + remainder * (base_step + 1) + (lane - remainder) * base_step;
    T sample = data[idx];
    sample = warp_bitonic32_t<T, Cmp>(sample, 32, lane);
    return ws_shfl_t(sample, 15);
}

// Three-way partition of data[seg_lo..seg_hi) around pivot into tmp, then
// copies back.  All lanes compute identical left_idx / right_idx (deterministic
// via __ballot_sync + __popc), so *out_mid_lo / *out_mid_hi are valid on every lane.
// All 32 lanes must call with identical arguments.
template<typename T, typename Cmp>
__device__ inline void warp_partition(
    T* data, T* tmp, int seg_lo, int seg_hi, T pivot,
    int* out_mid_lo, int* out_mid_hi, int lane)
{
    int seg_len = seg_hi - seg_lo;
    int left_idx = 0;
    int right_idx = 0;
    int full_blocks = seg_len / 32;
    int tail = seg_len - full_blocks * 32;

    for (int blk = 0; blk <= full_blocks; blk++) {
        int pos = seg_lo + blk * 32 + lane;
        bool active = (blk < full_blocks) || (lane < tail);

        T elem;
        if (active) elem = data[pos];

        int c = active ? Cmp::cmp(elem, pivot) : 0;
        bool is_less    = active && (c == -1);
        bool is_greater = active && (c ==  1);

        unsigned mask_less = __ballot_sync(WARP_MASK, is_less);
        int prefix_less = __popc(mask_less & ((1u << lane) - 1));
        if (is_less)
            tmp[seg_lo + left_idx + prefix_less] = elem;
        left_idx += __popc(mask_less);

        unsigned mask_gt = __ballot_sync(WARP_MASK, is_greater);
        int prefix_gt = __popc(mask_gt & ((1u << lane) - 1));
        if (is_greater)
            tmp[seg_hi - 1 - right_idx - prefix_gt] = elem;
        right_idx += __popc(mask_gt);

        __syncwarp();
    }

    // Copy left/right from tmp back to data; fill middle with pivot.
    int mid_lo = seg_lo + left_idx;
    int mid_hi = seg_hi - right_idx;
    for (int i = lane; i < left_idx; i += 32)
        data[seg_lo + i] = tmp[seg_lo + i];
    for (int i = lane; i < (mid_hi - mid_lo); i += 32)
        data[mid_lo + i] = pivot;
    for (int i = lane; i < right_idx; i += 32)
        data[mid_hi + i] = tmp[mid_hi + i];
    __syncwarp();

    *out_mid_lo = mid_lo;
    *out_mid_hi = mid_hi;
}

// Iterative warp quicksort on data[seg_lo..seg_hi) with explicit scratch.
// tmp must cover the full data range used by all recursive sub-problems.
// stack_lo/stack_hi: WS_MAX_STACK ints each.
// Returns 0 on success, 1 if stack overflowed.
// All 32 lanes must call with identical arguments.
template<typename T, typename Cmp>
__device__ inline int warp_sort_inner(
    T* data, T* tmp, int* stack_lo, int* stack_hi,
    int seg_lo, int seg_hi, int lane)
{
    int sp = 0;
    if (lane == 0) {
        stack_lo[0] = seg_lo;
        stack_hi[0] = seg_hi;
        sp = 1;
    }

    while (true) {
        int cur_sp = __shfl_sync(WARP_MASK, sp, 0);
        if (cur_sp <= 0) break;
        if (lane == 0) sp--;
        cur_sp--;
        int lo = stack_lo[cur_sp];
        int hi = stack_hi[cur_sp];
        int len = hi - lo;

        if (len <= 1) continue;

        if (len <= 32) {
            T val = (lane < len) ? data[lo + lane] : Cmp::sentinel();
            val = warp_bitonic32_t<T, Cmp>(val, len, lane);
            if (lane < len)
                data[lo + lane] = val;
            __syncwarp();
            continue;
        }

        T pivot = warp_pick_pivot<T, Cmp>(data, lo, len, lane);
        int mid_lo, mid_hi;
        warp_partition<T, Cmp>(data, tmp, lo, hi, pivot, &mid_lo, &mid_hi, lane);

        if (lane == 0) {
            int left_idx  = mid_lo - lo;
            int right_idx = hi - mid_hi;
            if (left_idx > 1) {
                if (sp >= WS_MAX_STACK) { sp = -1; }
                else { stack_lo[sp] = lo; stack_hi[sp] = mid_lo; sp++; }
            }
            if (sp >= 0 && right_idx > 1) {
                if (sp >= WS_MAX_STACK) { sp = -1; }
                else { stack_lo[sp] = mid_hi; stack_hi[sp] = hi; sp++; }
            }
        }
        __syncwarp();
        if (__shfl_sync(WARP_MASK, sp, 0) < 0) return 1;
    }
    return 0;
}

// Sort data[0..n) in-place.  Returns 0 on success, 1 if stack overflowed.
// scratch must be (n * sizeof(T) + WS_MAX_STACK * 2 * sizeof(int)) bytes.
// All 32 lanes must call with identical arguments.
template<typename T, typename Cmp>
__device__ inline int warp_sort_t(T* data, char* scratch, int n, int lane) {
    if (n <= 1) return 0;
    T*   tmp      = (T*)scratch;
    int* stack_lo = (int*)(scratch + n * (int)sizeof(T));
    int* stack_hi = stack_lo + WS_MAX_STACK;
    return warp_sort_inner<T, Cmp>(data, tmp, stack_lo, stack_hi, 0, n, lane);
}

// ============================================================================
// BtPoint32 — canonical definition, shared with hull_dandc.cuh
// ============================================================================

#ifndef BT_POINT32_DEFINED
#define BT_POINT32_DEFINED
struct BtPoint32 {
    int x, y, z, index;
};
#endif

// BtPoint32 comparator: lexicographic on (y, x, z, index)
struct BtPoint32Cmp {
    static __device__ inline int cmp(BtPoint32 a, BtPoint32 b) {
        if (a.y != b.y) return (a.y < b.y) ? -1 : 1;
        if (a.x != b.x) return (a.x < b.x) ? -1 : 1;
        if (a.z != b.z) return (a.z < b.z) ? -1 : 1;
        if (a.index != b.index) return (a.index < b.index) ? -1 : 1;
        return 0;
    }
    static __device__ inline BtPoint32 sentinel() {
        BtPoint32 s;
        s.x = 0x7fffffff; s.y = 0x7fffffff; s.z = 0x7fffffff; s.index = 0x7fffffff;
        return s;
    }
};

// ============================================================================
// Legacy API — warp_sort_bp32 and BtPoint32 helpers
// (Also needs C++ linkage since they call templates)
// ============================================================================

__device__ inline int ws_cmp(BtPoint32 a, BtPoint32 b) {
    return BtPoint32Cmp::cmp(a, b);
}
__device__ inline BtPoint32 ws_bitonic32(BtPoint32 val, int n, int lane) {
    return warp_bitonic32_t<BtPoint32, BtPoint32Cmp>(val, n, lane);
}

// Sort BtPoint32 array in-place (legacy wrapper).
// scratch: n * sizeof(BtPoint32) + WS_MAX_STACK * 2 * sizeof(int) bytes.
__device__ inline int warp_sort_bp32(BtPoint32* points, char* scratch, int n, int lane) {
    return warp_sort_t<BtPoint32, BtPoint32Cmp>(points, scratch, n, lane);
}
