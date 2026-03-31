// CPython extension module for GPU-accelerated convex decomposition.
// Provides beam search decomposition + Hausdorff distance computation.
// Uses Python Limited API (abi3) targeting Python 3.10+.
// Compiled by setuptools, not cmake.

#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x030A0000  // Python 3.10
#include <Python.h>
#include "beam.h"

// ---------------------------------------------------------------------------
// Module state: holds the GPU context
// ---------------------------------------------------------------------------

typedef struct {
    beam_ctx_t ctx;
} GpuState;

static GpuState g_state = { NULL };

// ---------------------------------------------------------------------------
// Helper: raise an error from beam_last_error
// ---------------------------------------------------------------------------

static PyObject* raise_error(beam_ctx_t ctx, int rc) {
    const char* msg = beam_last_error(ctx);
    PyErr_Format(PyExc_RuntimeError, "coacd_gpu error %d: %s", rc, msg ? msg : "unknown");
    return NULL;
}

#define REQUIRE_CTX() do { \
    if (!g_state.ctx) { \
        PyErr_SetString(PyExc_RuntimeError, "context not initialized; call init() first"); \
        return NULL; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// init(device=-1) -> None
// ---------------------------------------------------------------------------

static PyObject* py_init(PyObject* self, PyObject* args) {
    int device = -1;
    if (!PyArg_ParseTuple(args, "|i", &device))
        return NULL;

    if (g_state.ctx) {
        beam_destroy(g_state.ctx);
        g_state.ctx = NULL;
    }

    int rc = beam_init(&g_state.ctx, device);
    if (rc != 0) {
        return raise_error(g_state.ctx, rc);
    }
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// destroy() -> None
// ---------------------------------------------------------------------------

static PyObject* py_destroy(PyObject* self, PyObject* args) {
    if (g_state.ctx) {
        beam_destroy(g_state.ctx);
        g_state.ctx = NULL;
    }
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// run_v2(verts_ptr, n_verts, tris_ptr, n_tris,
//        hull_verts_ptr, n_hull_verts, hull_tris_ptr, n_hull_tris,
//        hull_volume, scratch_size,
//        beam_width, cuts_per_axis, threshold, rv_k,
//        max_parts, max_iterations, hausdorff_samples) -> int (num_parts)
// ---------------------------------------------------------------------------

static PyObject* py_run_v2(PyObject* self, PyObject* args) {
    unsigned long long verts_ptr, tris_ptr, hull_verts_ptr, hull_tris_ptr;
    int n_verts, n_tris, n_hull_verts, n_hull_tris;
    float hull_volume;
    unsigned long long scratch_size;
    int beam_width, cuts_per_axis, max_parts, max_iterations, hausdorff_samples;
    float threshold, rv_k;

    if (!PyArg_ParseTuple(args, "KiKiKiKifKiiffiii",
            &verts_ptr, &n_verts,
            &tris_ptr, &n_tris,
            &hull_verts_ptr, &n_hull_verts,
            &hull_tris_ptr, &n_hull_tris,
            &hull_volume, &scratch_size,
            &beam_width, &cuts_per_axis,
            &threshold, &rv_k,
            &max_parts, &max_iterations, &hausdorff_samples))
        return NULL;

    REQUIRE_CTX();

    beam_params_t params;
    params.beam_width = beam_width;
    params.cuts_per_axis = cuts_per_axis;
    params.threshold = threshold;
    params.rv_k = rv_k;
    params.max_parts = max_parts;
    params.max_iterations = max_iterations;
    params.hausdorff_samples = hausdorff_samples;

    int rc = beam_run_v2(g_state.ctx,
                         (const float*)(uintptr_t)verts_ptr, n_verts,
                         (const int*)(uintptr_t)tris_ptr, n_tris,
                         (const float*)(uintptr_t)hull_verts_ptr, n_hull_verts,
                         (const int*)(uintptr_t)hull_tris_ptr, n_hull_tris,
                         hull_volume, (size_t)scratch_size,
                         &params);
    if (rc != 0) return raise_error(g_state.ctx, rc);

    int num_parts = beam_get_num_parts(g_state.ctx);
    return PyLong_FromLong(num_parts);
}

// ---------------------------------------------------------------------------
// get_part_sizes(part_idx) -> (n_verts, n_tris)
// ---------------------------------------------------------------------------

static PyObject* py_get_part_sizes(PyObject* self, PyObject* args) {
    int part_idx;
    if (!PyArg_ParseTuple(args, "i", &part_idx))
        return NULL;
    REQUIRE_CTX();

    int n_verts = 0, n_tris = 0;
    beam_get_part(g_state.ctx, part_idx, NULL, &n_verts, NULL, &n_tris);

    PyObject* result = PyTuple_New(2);
    if (!result) return NULL;
    PyTuple_SetItem(result, 0, PyLong_FromLong(n_verts));
    PyTuple_SetItem(result, 1, PyLong_FromLong(n_tris));
    return result;
}

// ---------------------------------------------------------------------------
// get_part_info(part_idx) -> (rv_cost, hausdorff, mesh_volume, hull_volume)
// ---------------------------------------------------------------------------

static PyObject* py_get_part_info(PyObject* self, PyObject* args) {
    int part_idx;
    if (!PyArg_ParseTuple(args, "i", &part_idx))
        return NULL;
    REQUIRE_CTX();

    float rv_cost, hausdorff, mesh_volume, hull_volume;
    int rc = beam_get_part_info(g_state.ctx, part_idx,
                                &rv_cost, &hausdorff, &mesh_volume, &hull_volume);
    if (rc != 0) {
        PyErr_SetString(PyExc_IndexError, "part_idx out of range or no part info available");
        return NULL;
    }

    PyObject* result = PyTuple_New(4);
    if (!result) return NULL;
    PyTuple_SetItem(result, 0, PyFloat_FromDouble(rv_cost));
    PyTuple_SetItem(result, 1, PyFloat_FromDouble(hausdorff));
    PyTuple_SetItem(result, 2, PyFloat_FromDouble(mesh_volume));
    PyTuple_SetItem(result, 3, PyFloat_FromDouble(hull_volume));
    return result;
}

// ---------------------------------------------------------------------------
// get_part(part_idx, verts_ptr, n_verts, tris_ptr, n_tris) -> (n_verts, n_tris)
// ---------------------------------------------------------------------------

static PyObject* py_get_part(PyObject* self, PyObject* args) {
    int part_idx;
    unsigned long long verts_ptr, tris_ptr;
    int n_verts, n_tris;

    if (!PyArg_ParseTuple(args, "iKiKi",
            &part_idx, &verts_ptr, &n_verts, &tris_ptr, &n_tris))
        return NULL;
    REQUIRE_CTX();

    int rc = beam_get_part(g_state.ctx, part_idx,
                           (float*)(uintptr_t)verts_ptr, &n_verts,
                           (int*)(uintptr_t)tris_ptr, &n_tris);
    if (rc != 0 && rc != -1) return raise_error(g_state.ctx, rc);

    PyObject* result = PyTuple_New(2);
    if (!result) return NULL;
    PyTuple_SetItem(result, 0, PyLong_FromLong(n_verts));
    PyTuple_SetItem(result, 1, PyLong_FromLong(n_tris));
    return result;
}

// ---------------------------------------------------------------------------
// test_warp_sort(pts_ptr, total_pts, offsets_ptr, n_arrays) -> None
// ---------------------------------------------------------------------------

static PyObject* py_test_warp_sort(PyObject* self, PyObject* args) {
    unsigned long long pts_ptr, offsets_ptr;
    int total_pts, n_arrays;
    if (!PyArg_ParseTuple(args, "KiKi",
            &pts_ptr, &total_pts, &offsets_ptr, &n_arrays)) return NULL;
    REQUIRE_CTX();
    int rc = beam_test_warp_sort(g_state.ctx,
        (int*)(uintptr_t)pts_ptr, total_pts,
        (const int*)(uintptr_t)offsets_ptr, n_arrays);
    if (rc != 0) return raise_error(g_state.ctx, rc);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// batch_hull_volume(pts_ptr, total_pts, offsets_ptr, n_hulls, algo,
//                  max_pts_per_hull, vols_ptr, errs_ptr) -> None
// ---------------------------------------------------------------------------

static PyObject* py_batch_hull_volume(PyObject* self, PyObject* args) {
    unsigned long long pts_ptr, offsets_ptr, vols_ptr, errs_ptr;
    int total_pts, n_hulls, algo, max_pts_per_hull;
    if (!PyArg_ParseTuple(args, "KiKiiiKK",
            &pts_ptr, &total_pts, &offsets_ptr, &n_hulls,
            &algo, &max_pts_per_hull, &vols_ptr, &errs_ptr))
        return NULL;
    REQUIRE_CTX();
    int rc = beam_batch_hull_volume(g_state.ctx,
        (const float*)(uintptr_t)pts_ptr, total_pts,
        (const int*)  (uintptr_t)offsets_ptr, n_hulls,
        algo, max_pts_per_hull,
        (float*)(uintptr_t)vols_ptr,
        (int*)  (uintptr_t)errs_ptr);
    if (rc != 0) return raise_error(g_state.ctx, rc);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// batch_mesh_volume(verts_ptr, total_verts, tris_ptr, total_tris,
//                  tri_off_ptr, vert_off_ptr, n_meshes, vols_ptr) -> None
// ---------------------------------------------------------------------------

static PyObject* py_batch_mesh_volume(PyObject* self, PyObject* args) {
    unsigned long long verts_ptr, tris_ptr, toff_ptr, voff_ptr, vols_ptr;
    int total_verts, total_tris, n_meshes;
    if (!PyArg_ParseTuple(args, "KiKiKKiK",
            &verts_ptr, &total_verts,
            &tris_ptr,  &total_tris,
            &toff_ptr, &voff_ptr,
            &n_meshes, &vols_ptr))
        return NULL;
    REQUIRE_CTX();
    int rc = beam_batch_mesh_volume(g_state.ctx,
        (const float*)(uintptr_t)verts_ptr, total_verts,
        (const int*)  (uintptr_t)tris_ptr,  total_tris,
        (const int*)  (uintptr_t)toff_ptr,
        voff_ptr ? (const int*)(uintptr_t)voff_ptr : NULL,
        n_meshes,
        (float*)(uintptr_t)vols_ptr);
    if (rc != 0) return raise_error(g_state.ctx, rc);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// batch_hull_dandc_mesh(pts_ptr, total_pts, offsets_ptr, n_hulls,
//   max_pts_per_hull, max_hull_verts, max_hull_tris,
//   vols_ptr, errs_ptr, verts_ptr, tris_ptr, vc_ptr, tc_ptr) -> None
// ---------------------------------------------------------------------------

static PyObject* py_batch_hull_dandc_mesh(PyObject* self, PyObject* args) {
    unsigned long long pts_ptr, offsets_ptr, vols_ptr, errs_ptr;
    unsigned long long verts_ptr, tris_ptr, vc_ptr, tc_ptr;
    int total_pts, n_hulls, max_pts_per_hull, max_hull_verts, max_hull_tris;
    if (!PyArg_ParseTuple(args, "KiKiiiiKKKKKK",
            &pts_ptr, &total_pts, &offsets_ptr, &n_hulls,
            &max_pts_per_hull, &max_hull_verts, &max_hull_tris,
            &vols_ptr, &errs_ptr, &verts_ptr, &tris_ptr, &vc_ptr, &tc_ptr))
        return NULL;
    REQUIRE_CTX();
    int rc = beam_batch_hull_dandc_mesh(g_state.ctx,
        (const float*)(uintptr_t)pts_ptr, total_pts,
        (const int*)(uintptr_t)offsets_ptr, n_hulls,
        max_pts_per_hull, max_hull_verts, max_hull_tris,
        (float*)(uintptr_t)vols_ptr,
        (int*)(uintptr_t)errs_ptr,
        (float*)(uintptr_t)verts_ptr,
        (int*)(uintptr_t)tris_ptr,
        (int*)(uintptr_t)vc_ptr,
        (int*)(uintptr_t)tc_ptr);
    if (rc != 0) return raise_error(g_state.ctx, rc);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// test_termination(parts_ptr, n_parts, work_items_ptr, n_items, threshold) -> int
// ---------------------------------------------------------------------------
static PyObject* py_test_termination(PyObject* self, PyObject* args) {
    unsigned long long parts_ptr, wi_ptr;
    int n_parts, n_items;
    float threshold;
    if (!PyArg_ParseTuple(args, "KiKif", &parts_ptr, &n_parts, &wi_ptr, &n_items, &threshold))
        return NULL;
    REQUIRE_CTX();
    int result = beam_test_termination(g_state.ctx,
        (PartInfoV2*)(uintptr_t)parts_ptr, n_parts,
        (WorkItem*)(uintptr_t)wi_ptr, n_items,
        threshold);
    return PyLong_FromLong(result);
}

// ---------------------------------------------------------------------------
// test_hausdorff_parts(vp_ptr, n_vp, tp_ptr, n_tp,
//                      parts_ptr, n_parts, pidx_ptr, n_pidx, threshold) -> int
// ---------------------------------------------------------------------------
static PyObject* py_test_hausdorff_parts(PyObject* self, PyObject* args) {
    unsigned long long vp_ptr, tp_ptr, parts_ptr, pidx_ptr;
    int n_vp, n_tp, n_parts, n_pidx;
    float threshold;
    if (!PyArg_ParseTuple(args, "KiKiKiKif",
            &vp_ptr, &n_vp, &tp_ptr, &n_tp,
            &parts_ptr, &n_parts, &pidx_ptr, &n_pidx, &threshold))
        return NULL;
    REQUIRE_CTX();
    int rc = beam_test_hausdorff_parts(g_state.ctx,
        (const float*)(uintptr_t)vp_ptr, n_vp,
        (const int*)(uintptr_t)tp_ptr, n_tp,
        (PartInfoV2*)(uintptr_t)parts_ptr, n_parts,
        (const int*)(uintptr_t)pidx_ptr, n_pidx,
        threshold);
    return PyLong_FromLong(rc);
}

// ---------------------------------------------------------------------------
// test_expansion(vp_ptr, n_vp, tp_ptr, n_tp,
//                parts_ptr, n_parts, wi_ptr, n_items,
//                cuts_per_axis, rv_k, beam_width, scratch_size,
//                out_wi_ptr, out_parts_ptr, out_parts_cap,
//                out_vp_ptr, out_vert_cap, out_tp_ptr, out_tri_cap)
//   -> (n_out_parts, n_out_verts, n_out_tris, kernel_error)
// ---------------------------------------------------------------------------
static PyObject* py_test_expansion(PyObject* self, PyObject* args) {
    unsigned long long vp_ptr, tp_ptr, parts_ptr, wi_ptr;
    int n_vp, n_tp, n_parts, n_items;
    int cuts_per_axis, beam_width;
    float rv_k;
    unsigned long long scratch_size;
    unsigned long long out_wi_ptr, out_parts_ptr, out_vp_ptr, out_tp_ptr;
    int out_parts_cap, out_vert_cap, out_tri_cap;

    if (!PyArg_ParseTuple(args, "KiKiKiKiifiKKKiKiKi",
            &vp_ptr, &n_vp, &tp_ptr, &n_tp,
            &parts_ptr, &n_parts, &wi_ptr, &n_items,
            &cuts_per_axis, &rv_k, &beam_width, &scratch_size,
            &out_wi_ptr, &out_parts_ptr, &out_parts_cap,
            &out_vp_ptr, &out_vert_cap, &out_tp_ptr, &out_tri_cap))
        return NULL;
    REQUIRE_CTX();

    int out_parts_count = 0, out_vert_count = 0, out_tri_count = 0, kerr = 0;
    int rc = beam_test_expansion(g_state.ctx,
        (const float*)(uintptr_t)vp_ptr, n_vp,
        (const int*)(uintptr_t)tp_ptr, n_tp,
        (const PartInfoV2*)(uintptr_t)parts_ptr, n_parts,
        (const WorkItem*)(uintptr_t)wi_ptr, n_items,
        cuts_per_axis, rv_k, beam_width, (size_t)scratch_size,
        (WorkItem*)(uintptr_t)out_wi_ptr,
        (PartInfoV2*)(uintptr_t)out_parts_ptr, out_parts_cap,
        &out_parts_count,
        (float*)(uintptr_t)out_vp_ptr, out_vert_cap,
        (int*)(uintptr_t)out_tp_ptr, out_tri_cap,
        &out_vert_count, &out_tri_count, &kerr);
    if (rc != 0) return raise_error(g_state.ctx, rc);

    return Py_BuildValue("iiii", out_parts_count, out_vert_count, out_tri_count, kerr);
}

// ---------------------------------------------------------------------------
// test_plane_cut(verts_ptr, n_verts, tris_ptr, n_tris,
//                pa, pb, pc, pd,
//                out_v_ptr, out_v_cap, out_pt_ptr, out_pt_cap,
//                out_nt_ptr, out_nt_cap) -> (n_verts, n_pos_tris, n_neg_tris)
// ---------------------------------------------------------------------------

static PyObject* py_test_plane_cut(PyObject* self, PyObject* args) {
    unsigned long long vp, tp, ovp, optp, ontp;
    int nv, nt, ovc, optc, ontc;
    float pa, pb, pc, pd;

    if (!PyArg_ParseTuple(args, "KiKiffffKiKiKi",
            &vp, &nv, &tp, &nt,
            &pa, &pb, &pc, &pd,
            &ovp, &ovc, &optp, &optc, &ontp, &ontc))
        return NULL;

    REQUIRE_CTX();
    beam_set_plane_cut_ctx(g_state.ctx);

    int out_nv = 0, out_npt = 0, out_nnt = 0;
    int rc = beam_test_plane_cut(
        (const float*)(uintptr_t)vp, nv,
        (const int*)(uintptr_t)tp, nt,
        pa, pb, pc, pd,
        (float*)(uintptr_t)ovp, ovc,
        (int*)(uintptr_t)optp, optc,
        (int*)(uintptr_t)ontp, ontc,
        &out_nv, &out_npt, &out_nnt);
    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError, "plane_cut failed (buffer overflow or alloc error)");
        return NULL;
    }
    return Py_BuildValue("iii", out_nv, out_npt, out_nnt);
}

// ---------------------------------------------------------------------------
// Module definition (slot-based, abi3-compatible)
// ---------------------------------------------------------------------------

static PyMethodDef gpu_methods[] = {
    { "init",                   py_init,                   METH_VARARGS, "Initialize GPU context." },
    { "destroy",                py_destroy,                METH_NOARGS,  "Destroy GPU context." },
    { "run_v2",                 py_run_v2,                 METH_VARARGS, "Run V2 beam search (3-kernel)." },
    { "get_part_sizes",         py_get_part_sizes,         METH_VARARGS, "Get part vertex/triangle counts." },
    { "get_part",               py_get_part,               METH_VARARGS, "Copy part data to buffers." },
    { "get_part_info",          py_get_part_info,          METH_VARARGS, "Get part diagnostic info (rv, hausdorff, mesh_vol, hull_vol)." },
    { "test_warp_sort",             py_test_warp_sort,             METH_VARARGS, "Test warp sort BtPoint32." },
    { "batch_hull_volume",          py_batch_hull_volume,          METH_VARARGS, "Batch hull volume (three algorithms)." },
    { "batch_mesh_volume",          py_batch_mesh_volume,          METH_VARARGS, "Batch mesh volume (divergence theorem)." },
    { "batch_hull_dandc_mesh",      py_batch_hull_dandc_mesh,      METH_VARARGS, "Batch D&C hull volume + mesh extraction." },
    { "test_termination",           py_test_termination,           METH_VARARGS, "Test beam_termination kernel." },
    { "test_hausdorff_parts",       py_test_hausdorff_parts,       METH_VARARGS, "Test beam_hausdorff_parts kernel." },
    { "test_expansion",             py_test_expansion,             METH_VARARGS, "Test beam_expansion kernel." },
    { "test_plane_cut",             py_test_plane_cut,             METH_VARARGS, "CPU plane cut with cap triangulation." },
    { NULL, NULL, 0, NULL }
};

static int exec_gpu(PyObject* module) {
    PyModule_AddFunctions(module, gpu_methods);
    return 0;
}

static PyModuleDef_Slot gpu_slots[] = {
    { Py_mod_exec, (void*)exec_gpu },
    { 0, NULL }
};

static PyModuleDef gpu_module_def = {
    PyModuleDef_HEAD_INIT,
    "coacd_gpu._gpu",                             // module name
    "GPU-accelerated convex decomposition",        // docstring
    0,                                             // module state size
    NULL,                                          // methods (added via slot)
    gpu_slots,                                     // slots
    NULL, NULL, NULL                               // traverse, clear, free
};

PyMODINIT_FUNC PyInit__gpu(void) {
    return PyModuleDef_Init(&gpu_module_def);
}
