// error_codes.h — Host-side kernel error decoding.
//
// Includes the centralized error flag definitions from cuda/error_codes.cuh
// and provides kerr_decode() to turn a combined error word into a
// human-readable string listing all set error flags.

#ifndef ERROR_CODES_H
#define ERROR_CODES_H

#include "../cuda/error_codes.cuh"
#include <stdio.h>

// Decode a kernel error word into human-readable string.
// Returns number of characters written (excluding null terminator).
// Example output: "GPU error 0x81 [hull_dandc:warp_pool_oom, plane_cut:scratch_oom]"
static inline int kerr_decode(int err, char* buf, int bufsize) {
    if (!err) return 0;
    static const struct { int bit; const char* name; } table[] = {
        { KERR_BT_WARP_OOM,       "hull_dandc:warp_pool_oom" },
        { KERR_BT_SORT_STACK,     "hull_dandc:sort_stack_overflow" },
        { KERR_BT_DC_STACK,       "hull_dandc:dc_stack_overflow" },
        { KERR_BT_POOL_EXHAUST,   "hull_dandc:edge_pool_exhausted" },
        { KERR_BT_HEAP_TO_WARP,   "hull_dandc:heap_to_warp_oom" },
        { KERR_BT_HEAP_OUTPUT,    "hull_dandc:heap_output_oom" },
        { KERR_BT_EXTRACT_FAIL,   "hull_dandc:mesh_extract_fail" },
        { KERR_PC_SCRATCH_OOM,    "plane_cut:scratch_oom" },
        { KERR_PC_POOL_OOM,       "plane_cut:pool_oom" },
        { KERR_PC_SORT_ERR,       "plane_cut:sort_err" },
        { KERR_HD_SCRATCH_OOM,    "hausdorff:scratch_oom" },
        { KERR_HD_SORT_ERR,       "hausdorff:sort_err" },
        { KERR_DC_OOM,            "postprocess:oom" },
        { KERR_KDOP_SCRATCH_OOM,  "kdop_hull:scratch_oom" },
        { KERR_LA_OVERFLOW,       "lookahead:overflow" },
        { KERR_LA_SORT_OOM,       "lookahead:sort_oom" },
        { KERR_LA_SORT_STACK,     "lookahead:sort_stack_overflow" },
        { KERR_LA_EVAL_OOM,       "lookahead:eval_oom" },
        { KERR_MESH_INVALID,      "mesh_invalid" },
        { KERR_HEAP_OOM,          "heap:oom" },
        { KERR_HEAP_CORRUPT,      "heap:corrupt" },
        { KERR_HEAP_CORRUPT_PREV, "heap:corrupt_prev" },
    };
    int off = snprintf(buf, bufsize, "GPU error 0x%x [", err);
    int first = 1;
    for (int i = 0; i < (int)(sizeof(table) / sizeof(table[0])); i++) {
        if (err & table[i].bit) {
            off += snprintf(buf + off, bufsize - off, "%s%s",
                            first ? "" : ", ", table[i].name);
            first = 0;
        }
    }
    int unknown = err & ~KERR_ALL_BITS;
    if (unknown)
        off += snprintf(buf + off, bufsize - off, "%sunknown:0x%x",
                        first ? "" : ", ", unknown);
    off += snprintf(buf + off, bufsize - off, "]");
    return off;
}

#endif // ERROR_CODES_H
