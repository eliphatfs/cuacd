// warp_sort.cuh — Warp-cooperative quicksort for BtPoint32 arrays.
//
// All 32 lanes must call. Uses a global-memory workspace (same size as input
// + 4KB for stack) for partitioning scratch. Segments <= 32 elements are
// sorted via a bitonic network; larger segments use quicksort partitioning
// with a median-of-samples pivot.
//
// Defines BtPoint32 (the canonical definition used by hull_dandc.cuh too).
// Requires: hull_warp_common.cuh (WARP_SIZE, WARP_MASK)

#ifndef WARP_SORT_CUH
#define WARP_SORT_CUH

#include "hull_warp_common.cuh"

// BtPoint32 — canonical definition, shared with hull_dandc.cuh
#ifndef BT_POINT32_DEFINED
#define BT_POINT32_DEFINED
struct BtPoint32 {
    int x, y, z, index;
};
#endif

// ============================================================================
// Comparison — includes index as final tiebreaker for total ordering
// ============================================================================

__device__ inline int ws_cmp(BtPoint32 a, BtPoint32 b) {
    if (a.y != b.y) return (a.y < b.y) ? -1 : 1;
    if (a.x != b.x) return (a.x < b.x) ? -1 : 1;
    if (a.z != b.z) return (a.z < b.z) ? -1 : 1;
    if (a.index != b.index) return (a.index < b.index) ? -1 : 1;
    return 0;
}

// min/max for bitonic network
__device__ inline BtPoint32 ws_min(BtPoint32 a, BtPoint32 b) { return (ws_cmp(a, b) < 0) ? a : b; }
__device__ inline BtPoint32 ws_max(BtPoint32 a, BtPoint32 b) { return (ws_cmp(a, b) < 0) ? b : a; }

// ============================================================================
// Bitonic sort for <= 32 elements (all lanes participate)
// ============================================================================

__device__ inline BtPoint32 ws_bitonic32(BtPoint32 val, int n, int lane) {
    // Inactive lanes get a sentinel that sorts to the end
    BtPoint32 oob;
    oob.x = 0x7fffffff; oob.y = 0x7fffffff; oob.z = 0x7fffffff; oob.index = 0x7fffffff;
    if (lane >= n) val = oob;

    for (int k = 1; k <= 16; k <<= 1) {
        for (int j = k; j >= 1; j >>= 1) {
            BtPoint32 other;
            other.x = __shfl_xor_sync(WARP_MASK, val.x, j);
            other.y = __shfl_xor_sync(WARP_MASK, val.y, j);
            other.z = __shfl_xor_sync(WARP_MASK, val.z, j);
            other.index = __shfl_xor_sync(WARP_MASK, val.index, j);
            bool asc = ((lane & (k << 1)) == 0);
            if ((lane & j) == 0)
                val = asc ? ws_min(val, other) : ws_max(val, other);
            else
                val = asc ? ws_max(val, other) : ws_min(val, other);
        }
    }
    return val;
}

// ============================================================================
// Warp quicksort
// ============================================================================

#define WS_MAX_STACK 2048

// Sort points[0..n) in-place.
// scratch must be (n * sizeof(BtPoint32) + WS_MAX_STACK * 2 * sizeof(int)) bytes.
// All 32 lanes must call with identical arguments.
__device__ inline void warp_sort_bp32(BtPoint32* points, char* scratch, int n, int lane) {
    if (n <= 1) return;

    BtPoint32* tmp = (BtPoint32*)scratch;
    // Stack lives in global memory after the tmp array
    int* stack_lo = (int*)(scratch + n * (int)sizeof(BtPoint32));
    int* stack_hi = stack_lo + WS_MAX_STACK;

    // sp lives in lane-0 register, broadcast via shfl
    int sp = 0;
    if (lane == 0) {
        stack_lo[0] = 0;
        stack_hi[0] = n;
        sp = 1;
    }

    while (true) {
        // Broadcast sp from lane 0
        int cur_sp = __shfl_sync(WARP_MASK, sp, 0);
        if (cur_sp <= 0) break;
        // Pop — lane 0 decrements, all threads read stack from global mem
        if (lane == 0) sp--;
        cur_sp--;
        int seg_lo = stack_lo[cur_sp];
        int seg_hi = stack_hi[cur_sp];
        int seg_len = seg_hi - seg_lo;

        if (seg_len <= 1) continue;

        if (seg_len <= 32) {
            // Bitonic sort this small segment
            BtPoint32 oob;
            oob.x = 0x7fffffff; oob.y = 0x7fffffff; oob.z = 0x7fffffff; oob.index = 0x7fffffff;
            BtPoint32 val = (lane < seg_len) ? points[seg_lo + lane] : oob;
            val = ws_bitonic32(val, seg_len, lane);
            if (lane < seg_len)
                points[seg_lo + lane] = val;
            __syncwarp();
            continue;
        }

        // --- Pick pivot: sample 32 elements uniformly, bitonic sort, take median ---
        BtPoint32 sample;
        {
            int base_step = seg_len / 32;
            int remainder = seg_len - base_step * 32;
            int idx;
            if (lane < remainder)
                idx = seg_lo + lane * (base_step + 1);
            else
                idx = seg_lo + remainder * (base_step + 1) + (lane - remainder) * base_step;
            sample = points[idx];
        }
        sample = ws_bitonic32(sample, 32, lane);
        // Broadcast pivot (element 15) from lane 15
        BtPoint32 pivot;
        pivot.x = __shfl_sync(WARP_MASK, sample.x, 15);
        pivot.y = __shfl_sync(WARP_MASK, sample.y, 15);
        pivot.z = __shfl_sync(WARP_MASK, sample.z, 15);
        pivot.index = __shfl_sync(WARP_MASK, sample.index, 15);

        // --- Three-way partition into tmp ---
        // Items < pivot go left; items > pivot go right; items == pivot skipped.
        int left_idx = 0;
        int right_idx = 0;

        int full_blocks = seg_len / 32;
        int tail = seg_len - full_blocks * 32;

        for (int blk = 0; blk <= full_blocks; blk++) {
            int pos = seg_lo + blk * 32 + lane;
            bool active = (blk < full_blocks) || (lane < tail);

            BtPoint32 elem;
            if (active) elem = points[pos];

            int cmp = active ? ws_cmp(elem, pivot) : 0;
            bool is_less    = active && (cmp == -1);
            bool is_greater = active && (cmp == 1);

            // Write items < pivot from the left
            unsigned mask_less = __ballot_sync(WARP_MASK, is_less);
            int prefix_less = __popc(mask_less & ((1u << lane) - 1));
            if (is_less)
                tmp[seg_lo + left_idx + prefix_less] = elem;
            left_idx += __popc(mask_less);

            // Write items > pivot from the right
            unsigned mask_gt = __ballot_sync(WARP_MASK, is_greater);
            int prefix_gt = __popc(mask_gt & ((1u << lane) - 1));
            if (is_greater)
                tmp[seg_hi - 1 - right_idx - prefix_gt] = elem;
            right_idx += __popc(mask_gt);

            __syncwarp();
        }

        // --- Fill middle with pivot, copy left/right from tmp ---
        int mid_lo = seg_lo + left_idx;
        int mid_hi = seg_hi - right_idx;
        for (int i = lane; i < left_idx; i += 32)
            points[seg_lo + i] = tmp[seg_lo + i];
        for (int i = lane; i < (mid_hi - mid_lo); i += 32)
            points[mid_lo + i] = pivot;
        for (int i = lane; i < right_idx; i += 32)
            points[mid_hi + i] = tmp[mid_hi + i];
        __syncwarp();

        // --- Push sub-segments onto stack (lane 0 only) ---
        // [seg_lo, mid_lo) are < pivot, [mid_lo, mid_hi) are == pivot, [mid_hi, seg_hi) are > pivot.
        if (lane == 0) {
            if (left_idx > 1 && sp < WS_MAX_STACK) {
                stack_lo[sp] = seg_lo;
                stack_hi[sp] = mid_lo;
                sp++;
            }
            if (right_idx > 1 && sp < WS_MAX_STACK) {
                stack_lo[sp] = mid_hi;
                stack_hi[sp] = seg_hi;
                sp++;
            }
        }
        __syncwarp();
    }
}

#endif // WARP_SORT_CUH
