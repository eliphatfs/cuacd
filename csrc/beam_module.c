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
// run(vertices_ptr, n_verts, triangles_ptr, n_tris,
//     beam_width, cuts_per_axis, threshold, rv_k,
//     max_parts, max_iterations, hausdorff_samples) -> int (num_parts)
// ---------------------------------------------------------------------------

static PyObject* py_run(PyObject* self, PyObject* args) {
    unsigned long long verts_ptr, tris_ptr;
    int n_verts, n_tris;
    int beam_width, cuts_per_axis, max_parts, max_iterations, hausdorff_samples;
    float threshold, rv_k;

    if (!PyArg_ParseTuple(args, "KiKiiiffiii",
            &verts_ptr, &n_verts,
            &tris_ptr, &n_tris,
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

    int rc = beam_run(g_state.ctx,
                      (const float*)(uintptr_t)verts_ptr, n_verts,
                      (const int*)(uintptr_t)tris_ptr, n_tris,
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
// point_mesh_distances(points_ptr, n_points, verts_ptr, n_verts,
//                      tris_ptr, n_tris, dist_ptr) -> None
// ---------------------------------------------------------------------------

static PyObject* py_point_mesh_distances(PyObject* self, PyObject* args) {
    unsigned long long points_ptr, verts_ptr, tris_ptr, dist_ptr;
    int n_points, n_verts, n_tris;

    if (!PyArg_ParseTuple(args, "KiKiKiK",
            &points_ptr, &n_points,
            &verts_ptr, &n_verts,
            &tris_ptr, &n_tris,
            &dist_ptr))
        return NULL;
    REQUIRE_CTX();

    int rc = beam_point_mesh_distances(g_state.ctx,
        (const float*)(uintptr_t)points_ptr, n_points,
        (const float*)(uintptr_t)verts_ptr, n_verts,
        (const int*)(uintptr_t)tris_ptr, n_tris,
        (float*)(uintptr_t)dist_ptr);
    if (rc != 0) return raise_error(g_state.ctx, rc);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// hausdorff(sa_ptr, n_sa, va_ptr, n_va, ta_ptr, n_ta,
//           sb_ptr, n_sb, vb_ptr, n_vb, tb_ptr, n_tb) -> float
// ---------------------------------------------------------------------------

static PyObject* py_hausdorff(PyObject* self, PyObject* args) {
    unsigned long long sa_ptr, va_ptr, ta_ptr, sb_ptr, vb_ptr, tb_ptr;
    int n_sa, n_va, n_ta, n_sb, n_vb, n_tb;

    if (!PyArg_ParseTuple(args, "KiKiKiKiKiKi",
            &sa_ptr, &n_sa, &va_ptr, &n_va, &ta_ptr, &n_ta,
            &sb_ptr, &n_sb, &vb_ptr, &n_vb, &tb_ptr, &n_tb))
        return NULL;
    REQUIRE_CTX();

    float result;
    int rc = beam_hausdorff(g_state.ctx,
        (const float*)(uintptr_t)sa_ptr, n_sa,
        (const float*)(uintptr_t)va_ptr, n_va,
        (const int*)(uintptr_t)ta_ptr, n_ta,
        (const float*)(uintptr_t)sb_ptr, n_sb,
        (const float*)(uintptr_t)vb_ptr, n_vb,
        (const int*)(uintptr_t)tb_ptr, n_tb,
        &result);
    if (rc != 0) return raise_error(g_state.ctx, rc);
    return PyFloat_FromDouble((double)result);
}

// ---------------------------------------------------------------------------
// pairwise_hausdorff(samples_ptr, soff_ptr, verts_ptr, tris_ptr,
//                    toff_ptr, voff_ptr, n_parts, cost_ptr) -> None
// ---------------------------------------------------------------------------

static PyObject* py_pairwise_hausdorff(PyObject* self, PyObject* args) {
    unsigned long long samples_ptr, soff_ptr, verts_ptr, tris_ptr;
    unsigned long long toff_ptr, voff_ptr, cost_ptr;
    int n_parts;

    if (!PyArg_ParseTuple(args, "KKKKKKiK",
            &samples_ptr, &soff_ptr,
            &verts_ptr, &tris_ptr,
            &toff_ptr, &voff_ptr,
            &n_parts, &cost_ptr))
        return NULL;
    REQUIRE_CTX();

    int rc = beam_pairwise_hausdorff(g_state.ctx,
        (const float*)(uintptr_t)samples_ptr,
        (const int*)(uintptr_t)soff_ptr,
        (const float*)(uintptr_t)verts_ptr,
        (const int*)(uintptr_t)tris_ptr,
        (const int*)(uintptr_t)toff_ptr,
        (const int*)(uintptr_t)voff_ptr,
        n_parts,
        (float*)(uintptr_t)cost_ptr);
    if (rc != 0) return raise_error(g_state.ctx, rc);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// Module definition (slot-based, abi3-compatible)
// ---------------------------------------------------------------------------

static PyMethodDef gpu_methods[] = {
    { "init",                   py_init,                   METH_VARARGS, "Initialize GPU context." },
    { "destroy",                py_destroy,                METH_NOARGS,  "Destroy GPU context." },
    { "run",                    py_run,                    METH_VARARGS, "Run beam search decomposition." },
    { "get_part_sizes",         py_get_part_sizes,         METH_VARARGS, "Get part vertex/triangle counts." },
    { "get_part",               py_get_part,               METH_VARARGS, "Copy part data to buffers." },
    { "point_mesh_distances",   py_point_mesh_distances,   METH_VARARGS, "Compute point-to-mesh distances." },
    { "hausdorff",              py_hausdorff,              METH_VARARGS, "Compute Hausdorff distance." },
    { "pairwise_hausdorff",     py_pairwise_hausdorff,     METH_VARARGS, "Compute pairwise Hausdorff cost matrix." },
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
