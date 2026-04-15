// la_common.cuh — shared constants, error codes, and cost functions for
// the lookahead tree search decomposition kernels (la_expand.cu,
// la_refine.cu, la_lifecycle.cu).
#pragma once

#include "structs.cuh"
#include "error_codes.cuh"

// Sentinel refcount value: marks hull as heap-allocated by kdop_hull_block
// (should be freed with heap_free, not refcount-decremented).
// Valid refcounts are always aligned pointers; (int*)1 is never a valid address.
#define LA_REFCOUNT_HEAP ((int*)1)

// Error codes: see error_codes.cuh (KERR_LA_*)

// Lookahead uses full cost including Hausdorff — the lookahead tree structure
// handles non-monotonicity well (unlike beam search which fails to converge).
// Cost = max(k_rv * rv, hausdorff) where rv = cbrt(3/(4pi)*(hull_vol - mesh_vol)).
#define LA_PART_COST_K_RV 0.3f

static __device__ inline float la_part_cost(const Part& p) {
    const float pi = 3.14159265358979f;
    float rv = cbrtf((3.0f / (4.0f * pi)) * fmaxf(p.hull_vol - p.mesh_vol, 0.0f));
    return fmaxf(LA_PART_COST_K_RV * rv, p.hausdorff);
}

// rv-only cost (no Hausdorff) — used during tree exploration.
// rv = cbrt(3/(4pi)*(hull_vol - mesh_vol)) — equivalent sphere radius of
// the volume difference. Monotonically decreases with each good cut.
static __device__ inline float la_part_cost_rv(const Part& p) {
    const float pi = 3.14159265358979f;
    float rv = cbrtf((3.0f / (4.0f * pi)) * fmaxf(p.hull_vol - p.mesh_vol, 0.0f));
    return LA_PART_COST_K_RV * rv;
}
