"""
Build logic for cuacd.

One artifact is produced:
  cuacd._gpu — Native CPython extension for GPU convex decomposition,
               Hausdorff distance, and convex hull computation.

The CUDA fatbin is built separately (nvcc, any platform) and shipped as
package data next to the extension (cuacd/kernels.fatbin); it is pure GPU
code, so wheels for other platforms reuse a fatbin built anywhere else.
Pass CUACD_FATBIN=<path> to stage a prebuilt one instead of running nvcc.

Metadata lives in pyproject.toml.

Requires at build time: a C compiler, plus either a CUDA toolkit (nvcc) or a
prebuilt fatbin; the CUDA driver headers/imports come from the build dep
nvidia-cuda-runtime-cu12 (or any CUDA toolkit on CUDA_HOME).
At runtime: only libcuda (the driver). No CUDA toolkit or PyTorch needed.
"""

import os
import sys
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

try:
    from wheel.bdist_wheel import bdist_wheel

    class abi3_wheel(bdist_wheel):
        """Tag the wheel as abi3 (stable ABI), compatible with Python >= 3.10."""
        def get_tag(self):
            python, abi, plat = super().get_tag()
            if python.startswith("cp"):
                return "cp310", "abi3", plat
            return python, abi, plat

    _extra_cmdclass = {"bdist_wheel": abi3_wheel}
except ImportError:
    _extra_cmdclass = {}


_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------

def _find_cuda_home():
    for env in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(env)
        if val and os.path.isdir(val):
            return val
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    for default in ("/usr/local/cuda",):
        if os.path.isdir(default):
            return default
    raise RuntimeError(
        "Cannot find CUDA toolkit. Set CUDA_HOME or ensure nvcc is on PATH.")


def _cuda_stubs_dir(cuda_home):
    for sub in ("lib64/stubs", "lib/stubs", "lib/x64"):
        p = os.path.join(cuda_home, sub)
        if os.path.isdir(p):
            return p
    return os.path.join(cuda_home, "lib64", "stubs")


def _pip_cuda_runtime_dirs():
    """Headers + import lib from the pip package nvidia-cuda-runtime-cu12.

    That wheel ships cuda.h and (on Windows) cuda.lib, so the host extension
    can be built without a CUDA toolkit install.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec("nvidia.cuda_runtime")
    except (ImportError, ValueError):  # namespace parent absent
        return None, None
    if spec is None:
        return None, None
    base = None
    if getattr(spec, "submodule_search_locations", None):
        base = list(spec.submodule_search_locations)[0]
    elif spec.origin:
        base = os.path.dirname(spec.origin)
    if not base or not os.path.isfile(os.path.join(base, "include", "cuda.h")):
        return None, None
    lib = os.path.join(base, "lib", "x64" if sys.platform == "win32" else "")
    return os.path.join(base, "include"), lib


def _cuda_dirs():
    """(include_dir, lib_dir) for the CUDA driver API headers and import lib."""
    for env in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(env)
        if home and os.path.isfile(os.path.join(home, "include", "cuda.h")):
            return os.path.join(home, "include"), _cuda_stubs_dir(home)
    inc, lib = _pip_cuda_runtime_dirs()
    if inc:
        return inc, lib
    home = _find_cuda_home()
    return os.path.join(home, "include"), _cuda_stubs_dir(home)


# ---------------------------------------------------------------------------
# Fatbin generation
# ---------------------------------------------------------------------------

def _fatbin_gencode_flags():
    archs_str = os.environ.get("CUACD_GPU_ARCHS", "80;86;89;90")
    archs = [a.strip() for a in archs_str.replace(";", ",").split(",") if a.strip()]
    flags = []
    for a in archs:
        flags += [f"-gencode=arch=compute_{a},code=sm_{a}"]
    if archs:
        flags += [f"-gencode=arch=compute_{archs[-1]},code=compute_{archs[-1]}"]
    return flags


# Modules compiled separately; kernels.cu is the old monolithic entry point (unused).
_CUDA_MODULES = [
    "mm.cu",
    "kdop_const.cu",
    "test_warp_sort.cu",
    "test_hull_dandc.cu",
    "test_mesh_volume.cu",
    "test_plane_cut.cu",
    "la_expand.cu",
    "la_refine.cu",
    "la_lifecycle.cu",
    "la_concave.cu",
    "postprocess_merge.cu",
    "test_kdop_hull.cu",
    "test_hausdorff.cu",
    "test_postprocess.cu",
    "test_mesh_audit.cu",
    "pdmc_kernels.cu",
    "c2s_kernels.cu",
]


def _compile_fatbin(cuda_home, _unused_cu_file, fatbin_file, build_dir):
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    os.makedirs(build_dir, exist_ok=True)

    extra_defines = []
    if os.environ.get("CUACD_TRACK_EDGES"):
        extra_defines.append("-DTRACK_MAX_EDGE_PAIRS")
    if os.environ.get("CUACD_GPU_ARENAS"):
        extra_defines.append(f"-DHEAP_NUM_ARENAS={os.environ['CUACD_GPU_ARENAS']}")
    if os.environ.get("CUACD_BEAM_DEBUG"):
        extra_defines.append("-DCUACD_BEAM_DEBUG")
    if os.environ.get("CUACD_LEAK_PROBE"):
        extra_defines.append("-DCUACD_LEAK_PROBE")
    if os.environ.get("CUACD_SERIAL_MERGE"):
        extra_defines.append("-DBT_SERIAL_MERGE")
    if os.environ.get("CUACD_MEMCHECK"):
        extra_defines.append("-fdevice-sanitize=memcheck")

    gencode = _fatbin_gencode_flags()
    cuda_dir = os.path.join(_ROOT, "cuda")

    # Compile each module to a relocatable device object in parallel.
    # CUACD_PARALLEL controls max concurrent nvcc processes (default: all).
    max_jobs = int(os.environ.get("CUACD_PARALLEL", 0)) or len(_CUDA_MODULES)
    pending = []  # (Popen, name, obj)
    obj_files = []
    failed = []
    for name in _CUDA_MODULES:
        # Wait for a slot if at capacity.
        while len(pending) >= max_jobs:
            for i, (proc, pname, obj) in enumerate(pending):
                if proc.poll() is not None:
                    (failed if proc.returncode else obj_files).append(
                        pname if proc.returncode else obj)
                    pending.pop(i)
                    break
        src = os.path.join(cuda_dir, name)
        obj = os.path.join(build_dir, name.replace(".cu", ".o"))
        proc = subprocess.Popen([
            nvcc, src, "-rdc=true", "-dc", "-O3", "--use_fast_math",
            "--generate-line-info", "-Xptxas=-v",
            *extra_defines,
            *gencode, "-o", obj,
        ])
        pending.append((proc, name, obj))
    # Drain remaining.
    for proc, name, obj in pending:
        proc.wait()
        (failed if proc.returncode else obj_files).append(
            name if proc.returncode else obj)
    if failed:
        raise RuntimeError(f"nvcc failed for: {', '.join(failed)}")

    # Device-link all objects into a single fatbin.
    subprocess.check_call([
        nvcc, "--device-link", "--fatbin",
        *gencode, *obj_files, "-o", fatbin_file,
    ])


# ---------------------------------------------------------------------------
# Custom build_py / build_ext
# ---------------------------------------------------------------------------

_PKG_FATBIN = os.path.join(_ROOT, "cuacd", "kernels.fatbin")


def _stage_fatbin(obj_dir):
    """Make sure cuacd/kernels.fatbin exists next to the package.

    Idempotent, so it is called from both build_py (which runs before
    build_ext and is the step that collects package data) and build_ext.
    """
    prebuilt = os.environ.get("CUACD_FATBIN")
    if prebuilt:
        if not os.path.isfile(prebuilt):
            raise FileNotFoundError(f"CUACD_FATBIN not found: {prebuilt}")
        if os.path.abspath(prebuilt) != os.path.abspath(_PKG_FATBIN):
            shutil.copyfile(prebuilt, _PKG_FATBIN)
    elif not os.path.isfile(_PKG_FATBIN):
        os.makedirs(os.path.dirname(_PKG_FATBIN), exist_ok=True)
        _compile_fatbin(_find_cuda_home(), None, _PKG_FATBIN, obj_dir)


class CoacdBuildPy(build_py):
    def run(self):
        # package data is collected at this step, before build_ext has had a
        # chance to produce the fatbin — stage it first or it never lands in
        # the wheel.
        _stage_fatbin(os.path.join(self.build_lib, "..", "fatbin_objs"))
        super().run()


class CoacdBuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name == "cuacd._gpu":
            self._build_gpu(ext)
        else:
            super().build_extension(ext)

    def _build_gpu(self, ext):
        """Build the host CPython extension and stage the fatbin it loads.

        The fatbinary is pure GPU code (PTX + cubins), so it is built once,
        wherever nvcc is available, and shipped next to the extension as
        package data — no compiler on the host side has to parse tens of MB
        of device code, and Windows builds never touch nvcc.
        """
        _stage_fatbin(os.path.join(self.build_temp, "gpu_build"))

        include_dir, lib_dir = _cuda_dirs()
        ext.include_dirs = [
            os.path.join(_ROOT, "csrc"),   # heap.h
            include_dir,                    # cuda.h
        ]
        ext.library_dirs = [lib_dir]
        ext.libraries = ["cuda"]
        c_args = ["/std:c11"] if sys.platform == "win32" else ["-std=c11", "-D_POSIX_C_SOURCE=199309L"]
        if os.environ.get("CUACD_V2_DEBUG"):
            c_args.append("-DCUACD_V2_DEBUG=1")
        if os.environ.get("CUACD_GPU_ARENAS"):
            c_args.append(f"-DHEAP_NUM_ARENAS={os.environ['CUACD_GPU_ARENAS']}")
        ext.extra_compile_args = c_args

        build_ext.build_extension(self, ext)


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

_gpu_ext = Extension(
    name="cuacd._gpu",
    sources=[
        os.path.join("csrc", "module.c"),
        os.path.join("csrc", "heap.c"),
        os.path.join("csrc", "lookahead.c"),
        os.path.join("csrc", "test.c"),
        os.path.join("csrc", "postprocess.c"),
        os.path.join("csrc", "preprocess.c"),
    ],
    py_limited_api=True,
)

setup(
    packages=["cuacd"],
    ext_modules=[_gpu_ext],
    cmdclass={"build_ext": CoacdBuildExt, "build_py": CoacdBuildPy, **_extra_cmdclass},
    zip_safe=False,
)
