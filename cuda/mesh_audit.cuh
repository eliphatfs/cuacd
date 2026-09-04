// mesh_audit.cuh — GPU mesh topology audit (single-block / block-level).
//
// Reports whether a triangle mesh is watertight / edge-manifold / consistently
// oriented, as a bit-flag verdict. Designed as a cheap gate before optional
// remeshing (preprocess) — not a hot path.
//
// Contract (caller):
//   - One block of AUDIT_BLOCK cooperative threads.
//   - Caller heap-allocs edge-table + counter arrays from the scratch heap and
//     passes them in (sized via mesh_audit_capacity()).
//   - Output is an AuditVerdict bitmask in host, not used during other kernels.
#pragma once
#include "common.cuh"
#include "allocator.cuh"

#ifndef MAUDIT_BLOCK
#define MAUDIT_BLOCK 256
#endif

// Verdict bits (host-visible).
#define MAV_OK             0u
#define MAV_DEGEN_TRI      (1u<<0)   // at least one zero-area / repeated-index triangle
#define MAV_BAD_INDEX      (1u<<1)   // triangle index out of [0,nv)
#define MAV_OPEN_EDGE      (1u<<2)   // edge used by !=2 faces including 1 (boundary)
#define MAV_NONMANIFOLD    (1u<<3)   // edge used by >2 faces
#define MAV_FLIPPED        (1u<<4)   // an undirected edge consistently oriented wrong

// Edge-record: 16 b, stored in a slab; hashtable stores 32-bit slot handles.
struct AuditEdgeRec {
    unsigned int lo;        // min vertex id
    unsigned int hi;        // max vertex id
    unsigned int total;     // # incident triangles
    unsigned int fwd;       // # triangles that traverse (lo -> hi)
};

// Compute edge-slot capacity (next pow2 >= 2*max_edges, min 1024).
__host__ __device__ inline unsigned int mesh_audit_edge_capacity(int nt) {
    unsigned int need = (unsigned int)(3 * (unsigned long long)nt) * 2ull + 1ull;
    unsigned int cap = 1024u;
    while (cap < need) cap <<= 1;
    return cap;
}

// Find-or-insert a directed edge's undirected record in the open-addressed
// table. Returns a pointer to the record, or NULL on table-full.
__device__ inline AuditEdgeRec* maudit_find_or_insert(
    AuditEdgeRec* recs,        // [cap] records (uninit)
    unsigned int*  slots,      // [cap] 1-based rec index, 0 = empty
    unsigned int*  rec_count,  // next free record
    unsigned int   cap,
    unsigned int   a, unsigned int b)
{
    unsigned int lo = a < b ? a : b;
    unsigned int hi = a < b ? b : a;
    unsigned long long key = ((unsigned long long)lo << 32) | hi;
    unsigned int h = (unsigned int)((key * 0x9E3779B97F4A7C15ull) >> 32) & (cap - 1);
    for (unsigned int probe = 0; probe < cap; probe++) {
        unsigned int idx = (h + probe) & (cap - 1);
        unsigned int old = atomicCAS(&slots[idx], 0u, 0xFFFFFFFFu);  // claim empty slot
        if (old == 0xFFFFFFFFu) {
            // Another thread is inserting into THIS slot right now — spin until
            // it publishes (short window), do not skip ahead or we'd duplicate.
            do { __nanosleep(16); old = atomicAdd(&slots[idx], 0u); } while (old == 0xFFFFFFFFu);
        }
        if (old == 0u) {
            // we won an empty slot: fill a new record, then publish
            unsigned int rec = atomicAdd(rec_count, 1u);
            recs[rec].lo = lo; recs[rec].hi = hi;
            recs[rec].total = 0u; recs[rec].fwd = 0u;
            __threadfence();
            atomicExch(&slots[idx], rec + 1u);  // publish 1-based
            return &recs[rec];
        }
        const AuditEdgeRec* r = &recs[old - 1u];
        if (r->lo == lo && r->hi == hi)
            return (AuditEdgeRec*)r;           // match
        // else: collision → keep probing
    }
    return NULL;  // table full (should not happen if sized via capacity())
}

// ---------------------------------------------------------------------------
// mesh_audit_block: block-cooperative audit.
//
//   verts : float[nv*3]
//   tris  : int[nt*3]
//   scratch: recs[cap], slots[cap], counters[8] (zeroed by caller)
//   out_flags: single uint (written by thread 0)
// ---------------------------------------------------------------------------
__device__ inline void mesh_audit_block(
    const float* __restrict__ verts,
    const int*   __restrict__ tris,
    int nv, int nt,
    AuditEdgeRec* __restrict__ recs,
    unsigned int* __restrict__ slots,
    unsigned int* __restrict__ counters,   // [8]
    unsigned int cap,
    unsigned int* __restrict__ out_flags)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;

    // Phase A: per-triangle basic checks + edge accumulation.
    // counters: [0]=degen, [1]=bad_index
    if (tid == 0) { counters[0] = 0; counters[1] = 0; counters[2] = 0; }
    __syncthreads();
    for (int t = tid; t < nt; t += nthr) {
        int a = tris[t*3+0], b = tris[t*3+1], c = tris[t*3+2];
        bool bad = (unsigned)a >= (unsigned)nv || (unsigned)b >= (unsigned)nv ||
                   (unsigned)c >= (unsigned)nv;
        if (bad) { atomicAdd(&counters[1], 1u); continue; }
        if (a == b || b == c || a == c) atomicAdd(&counters[0], 1u);
        else {
            // exactly-zero area check
            float ax = verts[a*3], ay = verts[a*3+1], az = verts[a*3+2];
            float bx = verts[b*3], by = verts[b*3+1], bz = verts[b*3+2];
            float cx = verts[c*3], cy = verts[c*3+1], cz = verts[c*3+2];
            float abx = bx-ax, aby = by-ay, abz = bz-az;
            float acx = cx-ax, acy = cy-ay, acz = cz-az;
            float nx = aby*acz - abz*acy, ny = abz*acx - abx*acz, nz = abx*acy - aby*acx;
            if (nx*nx + ny*ny + nz*nz <= 0.0f) atomicAdd(&counters[0], 1u);
        }
        AuditEdgeRec* r;
        r = maudit_find_or_insert(recs, slots, &counters[2], cap, (unsigned)a, (unsigned)b);
        if (r) { atomicAdd(&r->total, 1u); if (a < b) atomicAdd(&r->fwd, 1u); }
        r = maudit_find_or_insert(recs, slots, &counters[2], cap, (unsigned)b, (unsigned)c);
        if (r) { atomicAdd(&r->total, 1u); if (b < c) atomicAdd(&r->fwd, 1u); }
        r = maudit_find_or_insert(recs, slots, &counters[2], cap, (unsigned)c, (unsigned)a);
        if (r) { atomicAdd(&r->total, 1u); if (c < a) atomicAdd(&r->fwd, 1u); }
    }
    __syncthreads();

    // Phase B: classify edges.
    // counters: [3]=open, [4]=nonmanifold, [5]=flipped
    unsigned int n_edges = counters[2];
    if (tid == 0) { counters[3] = 0; counters[4] = 0; counters[5] = 0; }
    __syncthreads();
    for (unsigned int e = tid; e < n_edges; e += nthr) {
        unsigned int tot = recs[e].total;
        unsigned int fwd = recs[e].fwd;
        unsigned int rev = tot - fwd;
        if (tot == 0) continue;
        if (tot == 1)      atomicAdd(&counters[3], 1u);           // open
        else if (tot > 2)  atomicAdd(&counters[4], 1u);           // non-manifold
        else if (fwd == 2 || rev == 2) atomicAdd(&counters[5], 1u);// flipped (same dir twice)
    }
    __syncthreads();

    if (tid == 0) {
        unsigned int f = MAV_OK;
        if (counters[0])  f |= MAV_DEGEN_TRI;
        if (counters[1])  f |= MAV_BAD_INDEX;
        if (counters[3])  f |= MAV_OPEN_EDGE;
        if (counters[4])  f |= MAV_NONMANIFOLD;
        if (counters[5])  f |= MAV_FLIPPED;
        *out_flags = f;
    }
}
