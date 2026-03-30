// warp_sort.cuh — Warp-cooperative quicksort for BtPoint32 arrays.
//
// All 32 lanes must call. Uses a global-memory workspace of the same size
// as the input for partitioning scratch. Segments <= 32 elements are
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
// Comparison
// ============================================================================

__device__ inline bool ws_less(BtPoint32 a, BtPoint32 b) {
    return (a.y < b.y) || ((a.y == b.y) && ((a.x < b.x) || ((a.x == b.x) && (a.z < b.z))));
}

// min/max for bitonic network
__device__ inline BtPoint32 ws_min(BtPoint32 a, BtPoint32 b) { return ws_less(a, b) ? a : b; }
__device__ inline BtPoint32 ws_max(BtPoint32 a, BtPoint32 b) { return ws_less(a, b) ? b : a; }

// ============================================================================
// Bitonic sort for <= 32 elements (all lanes participate)
// ============================================================================

__device__ inline BtPoint32 ws_bitonic32(BtPoint32 val, int n, int lane) {
    // Inactive lanes get a sentinel that sorts to the end
    BtPoint32 oob;
    oob.x = 0x7fffffff; oob.y = 0x7fffffff; oob.z = 0x7fffffff; oob.index = -1;
    if (lane >= n) val = oob;

    for (int k = 1; k <= 16; k <<= 1) {
        for (int j = k; j >= 1; j >>= 1) {
            // Exchange with partner
            BtPoint32 other;
            {
                // __shfl_xor on 4 ints
                int ox = __shfl_xor_sync(WARP_MASK, val.x, j);
                int oy = __shfl_xor_sync(WARP_MASK, val.y, j);
                int oz = __shfl_xor_sync(WARP_MASK, val.z, j);
                int oi = __shfl_xor_sync(WARP_MASK, val.index, j);
                other.x = ox; other.y = oy; other.z = oz; other.index = oi;
            }
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

// Max recursion depth — log2(N) for N up to millions; 64 is very generous
#define WS_MAX_STACK 64

// Sort points[0..n) in-place. tmp must be n elements of scratch space.
// All 32 lanes must call with identical arguments.
__device__ inline void warp_sort_bp32(BtPoint32* points, BtPoint32* tmp, int n, int lane) {
    if (n <= 1) return;

    // --- Stack lives in lane-0 registers, broadcast via shfl ---
    int stack_lo[WS_MAX_STACK];
    int stack_hi[WS_MAX_STACK];
    int sp = 0; // stack pointer (lane 0 only)

    if (lane == 0) {
        stack_lo[0] = 0;
        stack_hi[0] = n;
        sp = 1;
    }

    while (true) {
        // Broadcast stack pointer
        int cur_sp = __shfl_sync(WARP_MASK, sp, 0);
        if (cur_sp == 0) break;

        // Pop
        int seg_lo, seg_hi;
        if (lane == 0) {
            sp--;
            seg_lo = stack_lo[sp];
            seg_hi = stack_hi[sp];
        }
        seg_lo = __shfl_sync(WARP_MASK, seg_lo, 0);
        seg_hi = __shfl_sync(WARP_MASK, seg_hi, 0);
        int seg_len = seg_hi - seg_lo;

        if (seg_len <= 1) continue;

        if (seg_len <= 32) {
            // Bitonic sort this small segment
            BtPoint32 val;
            BtPoint32 oob;
            oob.x = 0x7fffffff; oob.y = 0x7fffffff; oob.z = 0x7fffffff; oob.index = -1;
            val = (lane < seg_len) ? points[seg_lo + lane] : oob;
            val = ws_bitonic32(val, seg_len, lane);
            if (lane < seg_len)
                points[seg_lo + lane] = val;
            __syncwarp();
            continue;
        }

        // --- Pick pivot: sample 32 elements uniformly, bitonic sort, take median ---
        BtPoint32 sample;
        {
            // Compute strided sample indices covering [seg_lo, seg_hi)
            // l % 32 threads step by (l/32+1), rest step by (l/32)
            int base_step = seg_len / 32;
            int remainder = seg_len - base_step * 32; // = seg_len % 32
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

        // --- Two-way partition into tmp ---
        // Items where ws_less(elem, pivot) go left; the rest go right.
        int left_idx = 0;
        int right_idx = 0;

        int full_blocks = seg_len / 32;
        int tail = seg_len - full_blocks * 32;

        for (int blk = 0; blk <= full_blocks; blk++) {
            int pos = seg_lo + blk * 32 + lane;
            bool active = (blk < full_blocks) || (lane < tail);

            BtPoint32 elem;
            if (active) elem = points[pos];

            bool is_less = active && ws_less(elem, pivot);
            bool is_geq  = active && !is_less;

            // Write items < pivot from the left
            unsigned mask_less = __ballot_sync(WARP_MASK, is_less);
            int prefix_less = __popc(mask_less & ((1u << lane) - 1));
            if (is_less)
                tmp[seg_lo + left_idx + prefix_less] = elem;
            left_idx += __popc(mask_less);

            // Write items >= pivot from the right
            unsigned mask_geq = __ballot_sync(WARP_MASK, is_geq);
            int prefix_geq = __popc(mask_geq & ((1u << lane) - 1));
            if (is_geq)
                tmp[seg_hi - 1 - right_idx - prefix_geq] = elem;
            right_idx += __popc(mask_geq);

            __syncwarp();
        }

        // --- Copy tmp back to points ---
        for (int i = lane; i < seg_len; i += 32)
            points[seg_lo + i] = tmp[seg_lo + i];
        __syncwarp();

        // --- Push sub-segments onto stack (lane 0) ---
        // [seg_lo, split) are < pivot, [split, seg_hi) are >= pivot.
        // Only push if strictly smaller than parent to guarantee progress.
        if (lane == 0) {
            int split = seg_lo + left_idx;
            if (split > seg_lo && sp < WS_MAX_STACK) {
                stack_lo[sp] = seg_lo;
                stack_hi[sp] = split;
                sp++;
            }
            if (seg_hi > split && (seg_hi - split) < seg_len && sp < WS_MAX_STACK) {
                stack_lo[sp] = split;
                stack_hi[sp] = seg_hi;
                sp++;
            }
        }
        __syncwarp();
    }
}

#endif // WARP_SORT_CUH
