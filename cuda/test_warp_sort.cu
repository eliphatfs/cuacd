// test_warp_sort.cu — Test kernel for warp_sort_bp32.
//
// Kernel: test_warp_sort_kernel
//   One warp per array. Sorts each array in-place.
//   points: packed [total_pts * 4] ints (x,y,z,index)
//   offsets: [n_arrays + 1]
//   scratch: per-warp scratch (n_pts * sizeof(BtPoint32) + stack overhead)
//   scratch_offsets: [n_arrays + 1] byte offsets into scratch

#include "warp_sort.cuh"

#define TEST_SORT_BLOCK 64

extern "C" __global__ void test_warp_sort_kernel(
    int*       __restrict__ points,           // packed BtPoint32 as int[4] per point
    const int* __restrict__ offsets,          // [n_arrays + 1] point offsets
    char*      __restrict__ scratch,          // total scratch buffer
    const int* __restrict__ scratch_offsets,  // [n_arrays + 1] byte offsets
    int                     n_arrays)
{
    int warps_per_block = TEST_SORT_BLOCK / WARP_SIZE;
    int warp_id = blockIdx.x * warps_per_block + (threadIdx.x / WARP_SIZE);
    int lane = threadIdx.x & (WARP_SIZE - 1);
    if (warp_id >= n_arrays) return;

    int start = offsets[warp_id];
    int count = offsets[warp_id + 1] - start;

    BtPoint32* pts = (BtPoint32*)(points + start * 4);
    char* scr = scratch + scratch_offsets[warp_id];

    warp_sort_bp32(pts, scr, count, lane);
}
