// Exercise the production candidate selector on the GPU without mesh cutting.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "la_axis_cut.cuh"

__global__ void select_candidates(int width, LaAxisCut* out) {
    int cut_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cut_idx < width) out[cut_idx] = la_axis_cut_candidate(width, cut_idx);
}

static bool cuda_ok(cudaError_t error) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    return false;
}

int main(int argc, char** argv) {
    int device_count = 0;
    cudaError_t device_error = cudaGetDeviceCount(&device_count);
    if (device_error == cudaErrorNoDevice || (device_error == cudaSuccess && device_count == 0)) return 77;
    if (!cuda_ok(device_error)) return 2;
    int only_width = argc > 1 ? std::atoi(argv[1]) : 0;
    if (argc > 1 && (only_width < 1 || only_width > 511)) return 2;
    LaAxisCut* device = nullptr;
    if (!cuda_ok(cudaMalloc(&device, 511 * sizeof(LaAxisCut)))) return 2;
    int checked = 0;
    for (int width = only_width ? only_width : 1; width <= (only_width ? only_width : 511); width++) {
        select_candidates<<<(width + 63) / 64, 64>>>(width, device);
        std::vector<LaAxisCut> candidates(width);
        if (!cuda_ok(cudaGetLastError()) || !cuda_ok(cudaMemcpy(
                candidates.data(), device, width * sizeof(LaAxisCut), cudaMemcpyDeviceToHost))) return 2;
        int counts[3] = {0, 0, 0};
        for (int idx = 0; idx < width; idx++) {
            LaAxisCut cut = candidates[idx];
            if (cut.axis < 0 || cut.axis >= 3 || cut.cuts_per_axis <= 0 ||
                    cut.slice < 0 || cut.slice >= cut.cuts_per_axis) {
                std::fprintf(stderr, "Invalid candidate: width=%d idx=%d axis=%d slice=%d count=%d\n",
                             width, idx, cut.axis, cut.slice, cut.cuts_per_axis);
                return 1;
            }
            // Every axis receives contiguous, unique, evenly spaced candidates.
            if (cut.slice != counts[cut.axis]++) return 1;
            float fraction = (cut.slice + 1.0f) / (cut.cuts_per_axis + 1.0f);
            if (!(fraction > 0.0f && fraction < 1.0f)) return 1;
            if (width % 3 == 0 && (cut.axis != idx / (width / 3) ||
                    cut.slice != idx % (width / 3) || cut.cuts_per_axis != width / 3)) return 1;
        }
        for (int axis = 0; axis < 3; axis++) {
            int expected_count = width / 3 + (axis < width % 3 ? 1 : 0);
            if (counts[axis] != expected_count) return 1;
        }
        if (width == 5) {
            const int expected_axes[5] = {0, 0, 1, 1, 2};
            const int expected_slices[5] = {0, 1, 0, 1, 0};
            const int expected_counts[5] = {2, 2, 2, 2, 1};
            for (int idx = 0; idx < 5; idx++) {
                LaAxisCut cut = candidates[idx];
                if (cut.axis != expected_axes[idx] || cut.slice != expected_slices[idx] ||
                        cut.cuts_per_axis != expected_counts[idx]) return 1;
            }
        }
        checked += width;
    }
    if (!cuda_ok(cudaFree(device))) return 2;
    std::printf("Checked %d axis-aligned candidates on the GPU\n", checked);
    return 0;
}
