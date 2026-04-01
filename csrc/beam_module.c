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
    { "test_warp_sort",         py_test_warp_sort,         METH_VARARGS, "Test warp sort BtPoint32." },
    { "batch_hull_volume",      py_batch_hull_volume,      METH_VARARGS, "Batch hull volume (D&C)." },
    { "batch_mesh_volume",      py_batch_mesh_volume,      METH_VARARGS, "Batch mesh volume (divergence theorem)." },
    { "batch_hull_dandc_mesh",  py_batch_hull_dandc_mesh,  METH_VARARGS, "Batch D&C hull volume + mesh extraction." },
    { "test_plane_cut",         py_test_plane_cut,         METH_VARARGS, "GPU plane cut with cap triangulation." },
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
