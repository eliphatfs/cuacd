// structs.cuh — shared device-side mesh/part/work-item structures.
#pragma once

// ============================================================================
// Mesh
// ============================================================================
struct Mesh {
    float* verts;      // interleaved xyz, nv * 3 floats
    int*   tris;       // triangle indices, nt * 3 ints
    int    nv;
    int    nt;
    int*   refcount;   // pointer into heap chunk; NULL if not heap-allocated
};

// ============================================================================
// Part — one convex piece with its hull and costs
// ============================================================================
struct Part {
    Mesh  mesh;
    Mesh  hull;
    float mesh_vol;
    float hull_vol;
    float hausdorff;
};

// ============================================================================
// PartPair — the two halves produced by a plane cut
// ============================================================================
struct PartPair {
    Part pos;  // positive half (plane normal side)
    Part neg;  // negative half
};

// ============================================================================
// WorkItem — a set of parts produced from one input mesh
// ============================================================================
#define WORK_ITEM_MAX_PARTS 512

struct WorkItem {
    Part parts[WORK_ITEM_MAX_PARTS];
    int  nparts;
};

// ============================================================================
// AlgoState — collection of work items
// ============================================================================
struct AlgoState {
    WorkItem* items;
    int       nitems;
};

// ============================================================================
// Lookahead search structs
// ============================================================================
#define LA_MAX_PARTS     16
#define LA_MAX_CUTTING   16
#define LA_MAX_DECOMP    1024
#define LA_MAX_LEVELS    4

struct LaWorkItem {
    Part parts[LA_MAX_PARTS];
    int  nparts;
    int  src_part_idx;     // index into cutting-parts list (level 0); -1 otherwise
    int  initial_cut_idx;  // which width-cut at level 0 produced this (0..width-1)
    float level_costs[LA_MAX_LEVELS]; // worst-part cost after each expansion level
    int  n_levels;         // number of levels recorded so far
    int  _pad[2];
};

struct LaDecompState {
    Part parts[LA_MAX_DECOMP];
    int  nparts;
    int  _pad;
};

struct LaEvalResult {
    float best_cost;       // minimum avg-worst-part cost across descendant leaves
    int   best_cut_idx;    // initial cut index achieving best_cost
    int   _pad[2];
};
