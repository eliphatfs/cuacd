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
