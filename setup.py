"""
Build logic for coacd_gpu.

Two extensions are built:
  1. coacd_gpu._native  — CMake-built shared library for Hausdorff/merge (ctypes, legacy)
  2. coacd_gpu._beam    — Native CPython extension for beam search decomposition

Both embed CUDA fatbins and link only against libcuda (driver API).
Metadata lives in pyproject.toml.

Requires at build time: cmake >= 3.24, CUDA toolkit (nvcc), a C compiler.
At runtime: only libcuda.so (the GPU driver). No CUDA toolkit or PyTorch needed.
"""

import os
import sys
import shutil
import subprocess

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


def _compile_fatbin(cuda_home, cu_file, fatbin_file, build_dir):
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    os.makedirs(build_dir, exist_ok=True)
    subprocess.check_call([
        nvcc, cu_file, "--fatbin", "-O3", "--use_fast_math",
        *_fatbin_gencode_flags(), "-o", fatbin_file,
    ], cwd=build_dir)


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
        if ext.name == "coacd_gpu._native":
            self._build_cmake(ext)
        elif ext.name == "coacd_gpu._beam":
            self._build_beam(ext)
        else:
            super().build_extension(ext)

    def _build_cmake(self, ext):
        """Build the CMake-based shared library (Hausdorff/merge kernels)."""
        cmake_src = os.path.join(_ROOT, "cuda")
        build_dir = os.path.join(self.build_temp, "cmake_build")
        os.makedirs(build_dir, exist_ok=True)

        ext_fullpath = os.path.abspath(self.get_ext_fullpath(ext.name))
        pkg_dir = os.path.dirname(ext_fullpath)
        os.makedirs(pkg_dir, exist_ok=True)

        cfg = "Release"
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={pkg_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={pkg_dir}",
            f"-DCOACD_GPU_CSRC_DIR={os.path.join(_ROOT, 'csrc')}",
        ]
        archs = os.environ.get("COACD_GPU_ARCHS")
        if archs:
            cmake_args.append(f"-DFATBIN_ARCHS={archs}")

        build_args = ["--config", cfg]
        if hasattr(self, "parallel") and self.parallel:
            build_args += [f"-j{self.parallel}"]

        subprocess.check_call(["cmake", cmake_src] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_dir)

        if not os.path.exists(ext_fullpath):
            lib_name = {"win32": "coacd_gpu.dll", "darwin": "libcoacd_gpu.dylib"
                        }.get(sys.platform, "libcoacd_gpu.so")
            lib_path = os.path.join(pkg_dir, lib_name)
            if os.path.exists(lib_path):
                os.symlink(lib_path, ext_fullpath)

    def _build_beam(self, ext):
        """Build beam search CPython extension with embedded fatbin."""
        cuda_home = _find_cuda_home()
        build_dir = os.path.join(self.build_temp, "beam_build")
        os.makedirs(build_dir, exist_ok=True)

        cu_file = os.path.join(_ROOT, "cuda", "beam_kernels.cu")
        fatbin_file = os.path.join(build_dir, "beam_kernels.fatbin")
        header_file = os.path.join(build_dir, "beam_kernels_fatbin.h")

        _compile_fatbin(cuda_home, cu_file, fatbin_file, build_dir)
        _fatbin_to_header(fatbin_file, header_file, "beam_kernels_fatbin")

        ext.include_dirs = [
            os.path.join(_ROOT, "csrc"),              # beam.h
            build_dir,                                 # beam_kernels_fatbin.h
            os.path.join(cuda_home, "include"),        # cuda.h
        ]
        ext.library_dirs = [_cuda_stubs_dir(cuda_home)]
        ext.libraries = ["cuda"]
        ext.extra_compile_args = (
            ["/std:c11"] if sys.platform == "win32" else ["-std=c11"]
        )

        build_ext.build_extension(self, ext)


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

_native_ext = Extension(
    name="coacd_gpu._native",
    sources=[],
    py_limited_api=True,
)

_beam_ext = Extension(
    name="coacd_gpu._beam",
    sources=[
        os.path.join("csrc", "beam_module.c"),
        os.path.join("csrc", "beam.c"),
    ],
    py_limited_api=True,
)

setup(
    packages=["coacd_gpu", "coacd_gpu.coacd"],
    ext_modules=[_native_ext, _beam_ext],
    cmdclass={"build_ext": CoacdBuildExt, **_extra_cmdclass},
    zip_safe=False,
)
