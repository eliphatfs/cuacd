"""
Build logic for coacd_gpu.

One extension is built:
  coacd_gpu._gpu — Native CPython extension for beam search decomposition,
                   Hausdorff distance, and pairwise merge cost.

Embeds CUDA fatbin and links only against libcuda (driver API).
Metadata lives in pyproject.toml.

Requires at build time: cmake >= 3.24, CUDA toolkit (nvcc), a C compiler.
At runtime: only libcuda.so (the GPU driver). No CUDA toolkit or PyTorch needed.
"""

import os
import sys
import shutil
import subprocess
import concurrent.futures

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

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


# ---------------------------------------------------------------------------
# Fatbin generation
# ---------------------------------------------------------------------------

def _fatbin_gencode_flags():
    archs_str = os.environ.get("COACD_GPU_ARCHS", "80;86;89;90")
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
    "test_warp_sort.cu",
    "test_hull_dandc.cu",
    "test_mesh_volume.cu",
    "test_plane_cut.cu",
    "beam.cu",
]


def _compile_fatbin(cuda_home, _unused_cu_file, fatbin_file, build_dir):
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    os.makedirs(build_dir, exist_ok=True)

    extra_defines = []
    if os.environ.get("COACD_TRACK_EDGES"):
        extra_defines.append("-DTRACK_MAX_EDGE_PAIRS")
    if os.environ.get("COACD_GPU_ARENAS"):
        extra_defines.append(f"-DHEAP_NUM_ARENAS={os.environ['COACD_GPU_ARENAS']}")
    if os.environ.get("COACD_BEAM_DEBUG"):
        extra_defines.append("-DCOACD_BEAM_DEBUG")

    gencode = _fatbin_gencode_flags()
    cuda_dir = os.path.join(_ROOT, "cuda")

    # Compile each module to a relocatable device object in parallel.
    obj_files = []
    def _compile_module(name):
        src = os.path.join(cuda_dir, name)
        obj = os.path.join(build_dir, name.replace(".cu", ".o"))
        subprocess.check_call([
            nvcc, src, "-rdc=true", "-dc", "-O3", "--use_fast_math",
            "--generate-line-info",
            *extra_defines,
            *gencode, "-o", obj,
        ])
        return obj

    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = {pool.submit(_compile_module, name): name for name in _CUDA_MODULES}
        for fut in concurrent.futures.as_completed(futures):
            obj_files.append(fut.result())  # raises on error

    # Device-link all objects into a single fatbin.
    subprocess.check_call([
        nvcc, "--device-link", "--fatbin",
        *gencode, *obj_files, "-o", fatbin_file,
    ])


def _fatbin_to_header(fatbin_file, header_file, symbol_name):
    with open(fatbin_file, "rb") as f:
        data = f.read()
    with open(header_file, "w") as f:
        f.write("// Auto-generated — do not edit.\n")
        f.write(f"static const unsigned char {symbol_name}[] = {{\n")
        for i, byte in enumerate(data):
            if i % 16 == 0:
                f.write("    ")
            f.write(f"0x{byte:02x},")
            f.write("\n" if i % 16 == 15 else " ")
        f.write("\n};\n")
        f.write(f"static const unsigned int {symbol_name}_len = {len(data)};\n")


# ---------------------------------------------------------------------------
# Custom build_ext
# ---------------------------------------------------------------------------

class CoacdBuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name == "coacd_gpu._gpu":
            self._build_gpu(ext)
        else:
            super().build_extension(ext)

    def _build_gpu(self, ext):
        """Build the GPU CPython extension with embedded fatbin."""
        cuda_home = _find_cuda_home()
        build_dir = os.path.join(self.build_temp, "gpu_build")
        os.makedirs(build_dir, exist_ok=True)

        cu_file = os.path.join(_ROOT, "cuda", "kernels.cu")
        fatbin_file = os.path.join(build_dir, "kernels.fatbin")
        header_file = os.path.join(build_dir, "kernels_fatbin.h")

        _compile_fatbin(cuda_home, cu_file, fatbin_file, build_dir)
        _fatbin_to_header(fatbin_file, header_file, "kernels_fatbin")

        ext.include_dirs = [
            os.path.join(_ROOT, "csrc"),              # beam.h
            build_dir,                                 # kernels_fatbin.h
            os.path.join(cuda_home, "include"),        # cuda.h
        ]
        ext.library_dirs = [_cuda_stubs_dir(cuda_home)]
        ext.libraries = ["cuda"]
        c_args = ["/std:c11"] if sys.platform == "win32" else ["-std=c11"]
        if os.environ.get("COACD_V2_DEBUG"):
            c_args.append("-DCOACD_V2_DEBUG=1")
        if os.environ.get("COACD_GPU_ARENAS"):
            c_args.append(f"-DHEAP_NUM_ARENAS={os.environ['COACD_GPU_ARENAS']}")
        ext.extra_compile_args = c_args

        build_ext.build_extension(self, ext)


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

_gpu_ext = Extension(
    name="coacd_gpu._gpu",
    sources=[
        os.path.join("csrc", "beam_module.c"),
        os.path.join("csrc", "beam.c"),
        os.path.join("csrc", "test_beam.c"),
    ],
    py_limited_api=True,
)

setup(
    packages=["coacd_gpu"],
    ext_modules=[_gpu_ext],
    cmdclass={"build_ext": CoacdBuildExt, **_extra_cmdclass},
    zip_safe=False,
)
