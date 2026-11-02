// Axis-aligned candidate indexing for the lookahead expansion.
#pragma once

struct LaAxisCut {
    int axis;
    int slice;
    int cuts_per_axis;
};

__host__ __device__ inline LaAxisCut la_axis_cut_candidate(int width, int cut_idx) {
    // Positive width, with cut_idx in [0, width). Keep each axis contiguous.
    int count = width / 3;
    int remainder = width % 3;
    int x_count = count + (remainder > 0);
    int y_count = count + (remainder > 1);
    if (cut_idx < x_count) return {0, cut_idx, x_count};
    if (cut_idx < x_count + y_count) return {1, cut_idx - x_count, y_count};
    return {2, cut_idx - x_count - y_count, count};
}
