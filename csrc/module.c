// CPython extension module for GPU-accelerated convex hull, mesh volume, and plane cut.
// Uses Python Limited API (abi3) targeting Python 3.10+.
// Compiled by setuptools, not cmake.

#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x030A0000  // Python 3.10
#include <Python.h>
#include <string.h>
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
// init(device=-1, pool_bytes=0) -> None
//   pool_bytes=0 → auto: 70% of free device memory at init time.
// ---------------------------------------------------------------------------

static PyObject* py_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"device", "pool_bytes", NULL};
    int device = -1;
    unsigned long long pool_bytes = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iK", kwlist,
                                     &device, &pool_bytes))
        return NULL;

    if (g_state.ctx) {
        beam_destroy(g_state.ctx);
        g_state.ctx = NULL;
    }

    int rc = beam_init(&g_state.ctx, device, (size_t)pool_bytes);
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
// heap_compact() -> None
//   Coalesce free blocks in both persistent heaps (output + scratch).
// ---------------------------------------------------------------------------

static PyObject* py_heap_compact(PyObject* self, PyObject* args) {
    REQUIRE_CTX();
    int rc = beam_heap_compact(g_state.ctx);
    if (rc != 0) return raise_error(g_state.ctx, rc);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// pool_usage() -> int
//   Return bytes consumed from the shared device pool (monotonically increases).
// ---------------------------------------------------------------------------

static PyObject* py_pool_usage(PyObject* self, PyObject* args) {
    REQUIRE_CTX();
    size_t usage = beam_pool_usage(g_state.ctx);
    return PyLong_FromSize_t(usage);
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
// hull_dandc(pts_ptr, total_pts, offsets_ptr, n_hulls,
//            max_pts_per_hull, max_hull_verts, max_hull_tris,
//            verts_ptr, tris_ptr, nv_ptr, nt_ptr, errors_ptr) -> None
// ---------------------------------------------------------------------------

static PyObject* py_hull_dandc(PyObject* self, PyObject* args) {
    unsigned long long pts_ptr, offsets_ptr, verts_ptr, tris_ptr;
    unsigned long long nv_ptr, nt_ptr, errors_ptr;
    int total_pts, n_hulls, max_pts, max_hv, max_ht;
    if (!PyArg_ParseTuple(args, "KiKiiiiKKKKK",
            &pts_ptr, &total_pts, &offsets_ptr, &n_hulls,
            &max_pts, &max_hv, &max_ht,
            &verts_ptr, &tris_ptr, &nv_ptr, &nt_ptr, &errors_ptr))
        return NULL;
    REQUIRE_CTX();
    int rc = beam_hull_dandc(g_state.ctx,
        (const float*)(uintptr_t)pts_ptr, total_pts,
        (const int*)  (uintptr_t)offsets_ptr, n_hulls,
        max_pts, max_hv, max_ht,
        (float*)(uintptr_t)verts_ptr,
        (int*)  (uintptr_t)tris_ptr,
        (int*)  (uintptr_t)nv_ptr,
        (int*)  (uintptr_t)nt_ptr,
        (int*)  (uintptr_t)errors_ptr);
    if (rc != 0) return raise_error(g_state.ctx, rc);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// test_mesh_volume(verts_ptr, n_verts, tris_ptr, n_tris, vol_ptr) -> None
// ---------------------------------------------------------------------------

static PyObject* py_test_mesh_volume(PyObject* self, PyObject* args) {
    unsigned long long verts_ptr, tris_ptr, vol_ptr;
    int n_verts, n_tris;
    if (!PyArg_ParseTuple(args, "KiKiK",
            &verts_ptr, &n_verts, &tris_ptr, &n_tris, &vol_ptr))
        return NULL;
    REQUIRE_CTX();
    int rc = beam_test_mesh_volume(g_state.ctx,
        (const float*)(uintptr_t)verts_ptr, n_verts,
        (const int*)  (uintptr_t)tris_ptr,  n_tris,
        (float*)      (uintptr_t)vol_ptr);
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
// test_plane_cut(verts_ptr, n_verts, tris_ptr, n_tris,
//               pa, pb, pc, pd,
//               out_pv_ptr, out_pv_cap,
//               out_pt_ptr, out_pt_cap,
//               out_nv_ptr, out_nv_cap,
//               out_nt_ptr, out_nt_cap)
//   -> (n_pv, n_pt, n_nv, n_nt)
// ---------------------------------------------------------------------------

static PyObject* py_test_plane_cut(PyObject* self, PyObject* args) {
    unsigned long long vp, tp, opvp, optp, onvp, ontp;
    int nv, nt, opvc, optc, onvc, ontc;
    float pa, pb, pc, pd;

    if (!PyArg_ParseTuple(args, "KiKiffffKiKiKiKi",
            &vp, &nv, &tp, &nt,
            &pa, &pb, &pc, &pd,
            &opvp, &opvc, &optp, &optc,
            &onvp, &onvc, &ontp, &ontc))
        return NULL;

    REQUIRE_CTX();

    int n_pv = 0, n_pt = 0, n_nv = 0, n_nt = 0;
    int rc = beam_test_plane_cut(g_state.ctx,
        (const float*)(uintptr_t)vp, nv,
        (const int*)  (uintptr_t)tp, nt,
        pa, pb, pc, pd,
        (float*)(uintptr_t)opvp, opvc,
        (int*)  (uintptr_t)optp, optc,
        (float*)(uintptr_t)onvp, onvc,
        (int*)  (uintptr_t)ontp, ontc,
        &n_pv, &n_pt, &n_nv, &n_nt);
    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError, "plane_cut failed");
        return NULL;
    }
    return Py_BuildValue("iiii", n_pv, n_pt, n_nv, n_nt);
}

// ---------------------------------------------------------------------------
// decompose(verts_ptr, nv, tris_ptr, nt,
//           hull_verts_ptr, hull_nv, hull_tris_ptr, hull_nt,
//           max_iters, cuts_per_axis, threshold, max_keep, verbose=0)
//   -> list of (verts_np, tris_np, mesh_vol, hull_vol, hausdorff)
//      one tuple per output part.
// verts/tris/hull_* are raw C pointers to contiguous float32/int32 arrays.
// ---------------------------------------------------------------------------

static PyObject* py_decompose(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {
        "verts_ptr", "nv", "tris_ptr", "nt",
        "hull_verts_ptr", "hull_nv", "hull_tris_ptr", "hull_nt",
        "max_iters", "cuts_per_axis", "threshold", "max_keep",
        "verbose", "debug", NULL
    };
    unsigned long long vp, tp, hvp, htp;
    int nv, nt, hull_nv, hull_nt;
    int max_iters, cuts_per_axis, max_keep;
    float threshold;
    int verbose = 0, debug = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "KiKiKiKiiifi|ii", kwlist,
            &vp, &nv, &tp, &nt,
            &hvp, &hull_nv, &htp, &hull_nt,
            &max_iters, &cuts_per_axis, &threshold, &max_keep,
            &verbose, &debug))
        return NULL;

    REQUIRE_CTX();

    struct beam_result result;
    memset(&result, 0, sizeof(result));

    int rc = beam_decompose(g_state.ctx,
        (const float*)(uintptr_t)vp,  nv,
        (const int*)  (uintptr_t)tp,  nt,
        (const float*)(uintptr_t)hvp, hull_nv,
        (const int*)  (uintptr_t)htp, hull_nt,
        max_iters, cuts_per_axis, threshold, max_keep,
        verbose, debug,
        &result);
    if (rc != 0) {
        beam_result_free(&result);
        return raise_error(g_state.ctx, rc);
    }

    // Build Python list of tuples, one per part.
    // Each tuple: (verts_memoryview, tris_memoryview, mesh_vol, hull_vol, hausdorff)
    // We copy into numpy-compatible bytes objects so the caller owns the data.
    PyObject* list = PyList_New(result.nparts);
    if (!list) { beam_result_free(&result); return NULL; }

    for (int i = 0; i < result.nparts; i++) {
        struct beam_part_result* p = &result.parts[i];
        Py_ssize_t vbytes = (Py_ssize_t)p->nv * 3 * sizeof(float);
        Py_ssize_t tbytes = (Py_ssize_t)p->nt * 3 * sizeof(int);

        PyObject* vbuf = PyBytes_FromStringAndSize((const char*)p->verts, vbytes);
        PyObject* tbuf = PyBytes_FromStringAndSize((const char*)p->tris,  tbytes);
        if (!vbuf || !tbuf) {
            Py_XDECREF(vbuf); Py_XDECREF(tbuf);
            Py_DECREF(list);
            beam_result_free(&result);
            return NULL;
        }

        PyObject* tup = Py_BuildValue("(OOiiff)",
            vbuf, tbuf,
            p->nv, p->nt,
            p->mesh_vol, p->hull_vol);
        Py_DECREF(vbuf);
        Py_DECREF(tbuf);
        if (!tup) {
            Py_DECREF(list);
            beam_result_free(&result);
            return NULL;
        }
        PyList_SetItem(list, i, tup);  /* steals ref; always succeeds for valid index */
    }

    beam_result_free(&result);
    return list;
}

// ---------------------------------------------------------------------------
// lookahead_decompose(verts_ptr, nv, tris_ptr, nt,
//                     hull_verts_ptr, hull_nv, hull_tris_ptr, hull_nt,
//                     max_iters, width, threshold,
//                     depth=2, quick_depth=1, max_n_cutting=16,
//                     verbose=0, debug=0)
//   -> list of (verts_bytes, tris_bytes, nv, nt, mesh_vol, hull_vol)
//      one tuple per output part.
// ---------------------------------------------------------------------------

static PyObject* py_lookahead_decompose(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {
        "verts_ptr", "nv", "tris_ptr", "nt",
        "hull_verts_ptr", "hull_nv", "hull_tris_ptr", "hull_nt",
        "max_iters", "width", "threshold",
        "depth", "quick_depth", "max_n_cutting",
        "verbose", "debug", NULL
    };
    unsigned long long vp, tp, hvp, htp;
    int nv, nt, hull_nv, hull_nt;
    int max_iters, width;
    float threshold;
    int depth = 2, quick_depth = 1, max_n_cutting = 16;
    int verbose = 0, debug = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "KiKiKiKiiifiii|ii", kwlist,
            &vp, &nv, &tp, &nt,
            &hvp, &hull_nv, &htp, &hull_nt,
            &max_iters, &width, &threshold,
            &depth, &quick_depth, &max_n_cutting,
            &verbose, &debug))
        return NULL;

    REQUIRE_CTX();

    struct beam_result result;
    memset(&result, 0, sizeof(result));

    int rc = lookahead_decompose(g_state.ctx,
        (const float*)(uintptr_t)vp,  nv,
        (const int*)  (uintptr_t)tp,  nt,
        (const float*)(uintptr_t)hvp, hull_nv,
        (const int*)  (uintptr_t)htp, hull_nt,
        max_iters, width, threshold,
        depth, quick_depth, max_n_cutting,
        verbose, debug,
        &result);
    if (rc != 0) {
        beam_result_free(&result);
        return raise_error(g_state.ctx, rc);
    }

    // Build Python list of tuples (same format as py_decompose)
    PyObject* list = PyList_New(result.nparts);
    if (!list) { beam_result_free(&result); return NULL; }

    for (int i = 0; i < result.nparts; i++) {
        struct beam_part_result* p = &result.parts[i];
        Py_ssize_t vbytes = (Py_ssize_t)p->nv * 3 * sizeof(float);
        Py_ssize_t tbytes = (Py_ssize_t)p->nt * 3 * sizeof(int);

        PyObject* vbuf = PyBytes_FromStringAndSize((const char*)p->verts, vbytes);
        PyObject* tbuf = PyBytes_FromStringAndSize((const char*)p->tris,  tbytes);
        if (!vbuf || !tbuf) {
            Py_XDECREF(vbuf); Py_XDECREF(tbuf);
            Py_DECREF(list);
            beam_result_free(&result);
            return NULL;
        }

        PyObject* tup = Py_BuildValue("(OOiiff)",
            vbuf, tbuf,
            p->nv, p->nt,
            p->mesh_vol, p->hull_vol);
        Py_DECREF(vbuf);
        Py_DECREF(tbuf);
        if (!tup) {
            Py_DECREF(list);
            beam_result_free(&result);
            return NULL;
        }
        PyList_SetItem(list, i, tup);
    }

    beam_result_free(&result);
    return list;
}

// ---------------------------------------------------------------------------
// kdop_hull(pts_ptr, total_pts, offsets_ptr, n_hulls, max_hv, max_ht,
//           verts_ptr, tris_ptr, nv_ptr, nt_ptr, vols_ptr, errors_ptr) -> None
// ---------------------------------------------------------------------------

static PyObject* py_kdop_hull(PyObject* self, PyObject* args) {
    unsigned long long pts_ptr, offsets_ptr, verts_ptr, tris_ptr;
    unsigned long long nv_ptr, nt_ptr, vols_ptr, errors_ptr;
    int total_pts, n_hulls, max_hv, max_ht;
    if (!PyArg_ParseTuple(args, "KiKiiiKKKKKK",
            &pts_ptr, &total_pts, &offsets_ptr, &n_hulls,
            &max_hv, &max_ht,
            &verts_ptr, &tris_ptr, &nv_ptr, &nt_ptr, &vols_ptr, &errors_ptr))
        return NULL;
    REQUIRE_CTX();
    int rc = beam_kdop_hull(g_state.ctx,
        (const float*)(uintptr_t)pts_ptr, total_pts,
        (const int*)  (uintptr_t)offsets_ptr, n_hulls,
        max_hv, max_ht,
        (float*)(uintptr_t)verts_ptr,
        (int*)  (uintptr_t)tris_ptr,
        (int*)  (uintptr_t)nv_ptr,
        (int*)  (uintptr_t)nt_ptr,
        (float*)(uintptr_t)vols_ptr,
        (int*)  (uintptr_t)errors_ptr);
    if (rc != 0) return raise_error(g_state.ctx, rc);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// test_hausdorff(hull_verts_ptr, hull_nv, hull_tris_ptr, hull_nt,
//               mesh_verts_ptr, mesh_nv, mesh_tris_ptr, mesh_nt,
//               out_ptr)  -> None
// ---------------------------------------------------------------------------

static PyObject* py_test_hausdorff(PyObject* self, PyObject* args) {
    unsigned long long hv_ptr, ht_ptr, mv_ptr, mt_ptr, out_ptr;
    int hull_nv, hull_nt, mesh_nv, mesh_nt;
    if (!PyArg_ParseTuple(args, "KiKiKiKiK",
            &hv_ptr, &hull_nv, &ht_ptr, &hull_nt,
            &mv_ptr, &mesh_nv, &mt_ptr, &mesh_nt,
            &out_ptr))
        return NULL;
    REQUIRE_CTX();
    int rc = beam_test_hausdorff(g_state.ctx,
        (const float*)(uintptr_t)hv_ptr, hull_nv,
        (const int*)  (uintptr_t)ht_ptr, hull_nt,
        (const float*)(uintptr_t)mv_ptr, mesh_nv,
        (const int*)  (uintptr_t)mt_ptr, mesh_nt,
        (float*)      (uintptr_t)out_ptr);
    if (rc != 0) return raise_error(g_state.ctx, rc);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// Module definition (slot-based, abi3-compatible)
// ---------------------------------------------------------------------------

static PyMethodDef gpu_methods[] = {
    { "init",               (PyCFunction)py_init,  METH_VARARGS | METH_KEYWORDS,
      "Initialize GPU context. init(device=-1, pool_bytes=0)\n"
      "pool_bytes=0 → auto: 70% of free device memory." },
    { "destroy",            py_destroy,            METH_NOARGS,  "Destroy GPU context." },
    { "heap_compact",       py_heap_compact,       METH_NOARGS,
      "Compact both persistent heaps (output + scratch). "
      "Call periodically to coalesce fragmented free blocks." },
    { "pool_usage",         py_pool_usage,         METH_NOARGS,
      "Return bytes consumed from the shared device pool (peak usage, monotonic)." },
    { "test_warp_sort",     py_test_warp_sort,     METH_VARARGS, "Test warp sort BtPoint32." },
    { "hull_dandc",         py_hull_dandc,         METH_VARARGS, "D&C convex hull mesh extraction." },
    { "test_mesh_volume",   py_test_mesh_volume,   METH_VARARGS, "Mesh volume (single mesh)." },
    { "batch_mesh_volume",  py_batch_mesh_volume,  METH_VARARGS, "Batch mesh volume (divergence theorem)." },
    { "test_plane_cut",     py_test_plane_cut,     METH_VARARGS, "GPU plane cut with cap triangulation." },
    { "kdop_hull",          py_kdop_hull,          METH_VARARGS, "k-DOP approximate hull mesh extraction." },
    { "test_hausdorff",     py_test_hausdorff,     METH_VARARGS, "Bidirectional Hausdorff distance." },
    { "decompose",          (PyCFunction)py_decompose, METH_VARARGS | METH_KEYWORDS,
      "decompose(verts_ptr, nv, tris_ptr, nt, hull_verts_ptr, hull_nv, hull_tris_ptr, hull_nt,\n"
      "          max_iters, cuts_per_axis, threshold, max_keep, verbose=0, debug=0)\n"
      "-> list of (verts_bytes, tris_bytes, nv, nt, mesh_vol, hull_vol) per part." },
    { "lookahead_decompose", (PyCFunction)py_lookahead_decompose, METH_VARARGS | METH_KEYWORDS,
      "lookahead_decompose(verts_ptr, nv, tris_ptr, nt, hull_verts_ptr, hull_nv, hull_tris_ptr, hull_nt,\n"
      "                    max_iters, width, threshold, depth=2, quick_depth=1, max_n_cutting=16, verbose=0, debug=0)\n"
      "-> list of (verts_bytes, tris_bytes, nv, nt, mesh_vol, hull_vol) per part." },
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
    "GPU-accelerated convex hull and plane cut",   // docstring
    0,                                             // module state size
    NULL,                                          // methods (added via slot)
    gpu_slots,                                     // slots
    NULL, NULL, NULL                               // traverse, clear, free
};

PyMODINIT_FUNC PyInit__gpu(void) {
    return PyModuleDef_Init(&gpu_module_def);
}
