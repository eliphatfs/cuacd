// Device-side data structures shared across kernel modules.
// Included by common.cuh (which then exposes these to all kernel modules).
// Must stay in sync with csrc/structs.h (host-side mirrors).

#ifndef STRUCTS_CUH
#define STRUCTS_CUH

struct PartInfo {
    int vert_offset, vert_count;
    int tri_offset, tri_count;
    float bbox[6];    // xmin,xmax,ymin,ymax,zmin,zmax
    float rv_cost;
};

struct BeamItem {
    int num_parts;
    int worst_part_idx;
    float worst_cost;
    int cut_count;
};

struct DevicePool {
    char*               base;
    unsigned long long* offset;
    unsigned long long  capacity;
};

#endif // STRUCTS_CUH
