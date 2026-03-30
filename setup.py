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
        "--generate-line-info",
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
        ext.extra_compile_args = (
            ["/std:c11"] if sys.platform == "win32" else ["-std=c11"]
        )

        build_ext.build_extension(self, ext)


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

_gpu_ext = Extension(
    name="coacd_gpu._gpu",
    sources=[
        os.path.join("csrc", "beam_module.c"),
        os.path.join("csrc", "beam.c"),
    ],
    py_limited_api=True,
)

setup(
    packages=["coacd_gpu"],
    ext_modules=[_gpu_ext],
    cmdclass={"build_ext": CoacdBuildExt, **_extra_cmdclass},
    zip_safe=False,
)
