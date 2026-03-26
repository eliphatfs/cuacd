// CPython extension module for GPU beam search convex decomposition.
// Uses Python Limited API (abi3) targeting Python 3.10+.
// Compiled by setuptools, not cmake.

#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x030A0000  // Python 3.10
#include <Python.h>
#include "beam.h"

// ---------------------------------------------------------------------------
// Module state: holds the beam context
// ---------------------------------------------------------------------------

typedef struct {
    beam_ctx_t ctx;
} BeamState;

static BeamState g_state = { NULL };

// ---------------------------------------------------------------------------
// Helper: raise an error from beam_last_error
// ---------------------------------------------------------------------------

static PyObject* raise_beam_error(beam_ctx_t ctx, int rc) {
    const char* msg = beam_last_error(ctx);
    PyErr_Format(PyExc_RuntimeError, "beam error %d: %s", rc, msg ? msg : "unknown");
    return NULL;
}

// ---------------------------------------------------------------------------
// beam.init(device=-1) -> None
// ---------------------------------------------------------------------------

static PyObject* py_beam_init(PyObject* self, PyObject* args) {
    int device = -1;
    if (!PyArg_ParseTuple(args, "|i", &device))
        return NULL;

    if (g_state.ctx) {
        beam_destroy(g_state.ctx);
        g_state.ctx = NULL;
    }

    int rc = beam_init(&g_state.ctx, device);
    if (rc != 0) {
        return raise_beam_error(g_state.ctx, rc);
    }
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// beam.destroy() -> None
// ---------------------------------------------------------------------------

static PyObject* py_beam_destroy(PyObject* self, PyObject* args) {
    if (g_state.ctx) {
        beam_destroy(g_state.ctx);
        g_state.ctx = NULL;
    }
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// beam.run(vertices_ptr, n_verts, triangles_ptr, n_tris,
//          beam_width, cuts_per_axis, threshold, rv_k,
//          max_parts, max_iterations, hausdorff_samples) -> int (num_parts)
//
// vertices_ptr / triangles_ptr are buffer data pointers (unsigned long long).
// ---------------------------------------------------------------------------

static PyObject* py_beam_run(PyObject* self, PyObject* args) {
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

    if (!g_state.ctx) {
        PyErr_SetString(PyExc_RuntimeError, "beam context not initialized; call init() first");
        return NULL;
    }

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
    if (rc != 0) {
        return raise_beam_error(g_state.ctx, rc);
    }

    int num_parts = beam_get_num_parts(g_state.ctx);
    return PyLong_FromLong(num_parts);
}

// ---------------------------------------------------------------------------
// beam.get_part_sizes(part_idx) -> (n_verts, n_tris)
// ---------------------------------------------------------------------------

static PyObject* py_beam_get_part_sizes(PyObject* self, PyObject* args) {
    int part_idx;
    if (!PyArg_ParseTuple(args, "i", &part_idx))
        return NULL;

    if (!g_state.ctx) {
        PyErr_SetString(PyExc_RuntimeError, "beam context not initialized");
        return NULL;
    }

    int n_verts = 0, n_tris = 0;
    beam_get_part(g_state.ctx, part_idx, NULL, &n_verts, NULL, &n_tris);

    PyObject* result = PyTuple_New(2);
    if (!result) return NULL;
    PyTuple_SetItem(result, 0, PyLong_FromLong(n_verts));
    PyTuple_SetItem(result, 1, PyLong_FromLong(n_tris));
    return result;
}

// ---------------------------------------------------------------------------
// beam.get_part(part_idx, verts_ptr, n_verts, tris_ptr, n_tris) -> (n_verts, n_tris)
//
// Writes data into caller-provided buffers (numpy array data pointers).
// ---------------------------------------------------------------------------

static PyObject* py_beam_get_part(PyObject* self, PyObject* args) {
    int part_idx;
    unsigned long long verts_ptr, tris_ptr;
    int n_verts, n_tris;

    if (!PyArg_ParseTuple(args, "iKiKi",
            &part_idx, &verts_ptr, &n_verts, &tris_ptr, &n_tris))
        return NULL;

    if (!g_state.ctx) {
        PyErr_SetString(PyExc_RuntimeError, "beam context not initialized");
        return NULL;
    }

    int rc = beam_get_part(g_state.ctx, part_idx,
                           (float*)(uintptr_t)verts_ptr, &n_verts,
                           (int*)(uintptr_t)tris_ptr, &n_tris);
    if (rc != 0 && rc != -1) {
        return raise_beam_error(g_state.ctx, rc);
    }

    PyObject* result = PyTuple_New(2);
    if (!result) return NULL;
    PyTuple_SetItem(result, 0, PyLong_FromLong(n_verts));
    PyTuple_SetItem(result, 1, PyLong_FromLong(n_tris));
    return result;
}

// ---------------------------------------------------------------------------
// Module definition (slot-based, abi3-compatible)
// ---------------------------------------------------------------------------

static PyMethodDef beam_methods[] = {
    { "init",           py_beam_init,           METH_VARARGS, "Initialize beam context." },
    { "destroy",        py_beam_destroy,        METH_NOARGS,  "Destroy beam context." },
    { "run",            py_beam_run,            METH_VARARGS, "Run beam search decomposition." },
    { "get_part_sizes", py_beam_get_part_sizes, METH_VARARGS, "Get part vertex/triangle counts." },
    { "get_part",       py_beam_get_part,       METH_VARARGS, "Copy part data to buffers." },
    { NULL, NULL, 0, NULL }
};

static int exec_beam(PyObject* module) {
    PyModule_AddFunctions(module, beam_methods);
    return 0;
}

static PyModuleDef_Slot beam_slots[] = {
    { Py_mod_exec, (void*)exec_beam },
    { 0, NULL }
};

static PyModuleDef beam_module_def = {
    PyModuleDef_HEAD_INIT,
    "coacd_gpu._beam",                          // module name
    "GPU beam search convex decomposition",      // docstring
    0,                                           // module state size
    NULL,                                        // methods (added via slot)
    beam_slots,                                  // slots
    NULL, NULL, NULL                             // traverse, clear, free
};

PyMODINIT_FUNC PyInit__beam(void) {
    return PyModuleDef_Init(&beam_module_def);
}
