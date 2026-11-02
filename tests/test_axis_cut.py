"""Regression tests for the production CUDA axis-aligned candidate selector."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def axis_cut_executable(tmp_path_factory):
    """Compile the GPU probe against the same helper used by la_expand."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA toolkit is required to compile the axis-cut GPU probe")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path_factory.mktemp("axis_cut") / "axis_cut_candidates"
    subprocess.run([nvcc, "-std=c++17", "-arch=native", "-lineinfo",
                    "-I", str(root / "cuda"), str(root / "tests/axis_cut_candidates.cu"),
                    "-o", str(executable)], check=True, capture_output=True, text=True)
    return executable


def run_probe(executable, *args):
    """Require valid GPU candidates, skipping only if no CUDA device exists."""
    result = subprocess.run([str(executable), *args], capture_output=True, text=True, timeout=30)
    if result.returncode == 77:
        pytest.skip("No CUDA device is available")
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_default_deeper_width(axis_cut_executable):
    """Map the default width2=5 to two X, two Y and one Z candidate."""
    assert "Checked 5 axis-aligned candidates" in run_probe(axis_cut_executable, "5")


def test_positive_widths_1_to_511(axis_cut_executable):
    """Cover widths 1 through 511 and preserve divisible-by-three layouts."""
    assert "Checked 130816 axis-aligned candidates" in run_probe(axis_cut_executable)
