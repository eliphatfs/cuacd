"""
Build logic for coacd_gpu.

Builds the C shared library (with embedded CUDA fatbin) via CMake.
Metadata lives in pyproject.toml; this file handles the CMake build step
and abi3 wheel tagging.

Requires at build time: cmake >= 3.24, CUDA toolkit (nvcc), a C compiler.
At runtime: only libcuda.so (the GPU driver). No CUDA toolkit or PyTorch needed.
"""

import os
import sys
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

try:
    from wheel.bdist_wheel import bdist_wheel

    class abi3_wheel(bdist_wheel):
        """Tag the wheel as abi3 (stable ABI), compatible with Python >= 3.9."""
        def get_tag(self):
            python, abi, plat = super().get_tag()
            if python.startswith("cp"):
                return "cp39", "abi3", plat
            return python, abi, plat

    _extra_cmdclass = {"bdist_wheel": abi3_wheel}
except ImportError:
    _extra_cmdclass = {}


class CMakeBuildExt(build_ext):
    """Build the C shared library via CMake, then let setuptools find it."""

    def build_extension(self, ext):
        source_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(self.build_temp, "cmake_build")
        os.makedirs(build_dir, exist_ok=True)

        # Place libcoacd_gpu.so next to python/__init__.py
        ext_fullpath = os.path.abspath(self.get_ext_fullpath(ext.name))
        pkg_dir = os.path.dirname(ext_fullpath)
        os.makedirs(pkg_dir, exist_ok=True)

        cfg = "Release"
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={pkg_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={pkg_dir}",
        ]

        # Allow overriding SM architectures via env var
        archs = os.environ.get("COACD_GPU_ARCHS")
        if archs:
            cmake_args.append(f"-DFATBIN_ARCHS={archs}")

        build_args = ["--config", cfg]
        if hasattr(self, "parallel") and self.parallel:
            build_args += [f"-j{self.parallel}"]

        subprocess.check_call(
            ["cmake", source_dir] + cmake_args,
            cwd=build_dir)
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_dir)

        # setuptools expects the dummy extension file to exist;
        # create it as a symlink to libcoacd_gpu.so
        if not os.path.exists(ext_fullpath):
            lib_name = "libcoacd_gpu.so"
            if sys.platform == "win32":
                lib_name = "coacd_gpu.dll"
            elif sys.platform == "darwin":
                lib_name = "libcoacd_gpu.dylib"
            lib_path = os.path.join(pkg_dir, lib_name)
            if os.path.exists(lib_path):
                os.symlink(lib_path, ext_fullpath)


# Dummy extension to trigger build_ext. Py_LIMITED_API enables abi3 tagging.
_dummy_ext = Extension(
    name="coacd_gpu._native",
    sources=[],
    py_limited_api=True,
)

setup(
    packages=["coacd_gpu", "coacd_gpu.coacd"],
    package_dir={"coacd_gpu": "python", "coacd_gpu.coacd": "python/coacd"},
    ext_modules=[_dummy_ext],
    cmdclass={"build_ext": CMakeBuildExt, **_extra_cmdclass},
    zip_safe=False,
)
