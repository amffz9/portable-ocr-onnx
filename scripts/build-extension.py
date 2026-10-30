#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified from nemotron-ocr 2.0.2 (NVIDIA): CPU-only CppExtension build when CUDA is
# unavailable, per-platform OpenMP/AVX2 flags.


import os
import platform
import sys
import shutil
import subprocess
from glob import glob

from pathlib import Path

def _parse_arch_list(arch_list: str) -> list[str]:
    """Parse TORCH_CUDA_ARCH_LIST-formatted string into nvcc -gencode flags.

    Accepts tokens like "7.5", "8.6", "8.6+PTX", "86", "90+PTX" separated by
    spaces, semicolons, or commas.
    """
    tokens = arch_list.replace(";", " ").replace(",", " ").split()
    flags: list[str] = []
    for token in tokens:
        with_ptx = token.endswith("+PTX")
        clean = token[:-4] if with_ptx else token
        clean = clean.replace(".", "")
        if not clean.isdigit():
            continue
        mm = clean
        flags.append(f"-gencode=arch=compute_{mm},code=sm_{mm}")
        if with_ptx:
            flags.append(f"-gencode=arch=compute_{mm},code=compute_{mm}")
    return flags


def _detect_arch_list() -> list[str]:
    """Best-effort detection of a single architecture if a GPU is visible.

    Falls back to a conservative list (sm_75 and sm_86) when detection isn't possible
    (common during Docker image build).
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            mm = f"{major}{minor}"
            return [f"-gencode=arch=compute_{mm},code=sm_{mm}"]
    except Exception:
        pass

    # Fallback: reasonably modern GPUs supported by CUDA 12.x toolchains
    return [
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_86,code=sm_86",
    ]


def _cuda_requested() -> bool:
    """Build the CUDA extension only if nvcc is usable and not explicitly disabled.

    NEMOTRON_OCR_CPU_ONLY=1 forces a CPU-only build even when nvcc is present.
    """
    if os.environ.get("NEMOTRON_OCR_CPU_ONLY", "0") == "1":
        return False
    try:
        import torch  # type: ignore
        from torch.utils.cpp_extension import CUDA_HOME  # type: ignore
    except Exception:
        return False
    if torch.version.cuda is None:  # CPU-only (or ROCm) torch can't build a CUDAExtension
        return False
    if CUDA_HOME is None:
        return False
    nvcc = Path(CUDA_HOME) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    return nvcc.exists() or shutil.which("nvcc") is not None


def _openmp_flags() -> tuple[list[str], list[str]]:
    """Return (compile_flags, link_flags) for OpenMP on this platform.

    - Linux/GCC: -fopenmp.
    - macOS/Apple clang: no built-in OpenMP. Uses Homebrew libomp
      (`brew install libomp`) via -Xpreprocessor -fopenmp. Set
      NEMOTRON_OCR_LIBOMP_PREFIX to override the prefix (default: `brew --prefix libomp`,
      then /opt/homebrew/opt/libomp, then /usr/local/opt/libomp).
    - NEMOTRON_OCR_NO_OPENMP=1 builds without OpenMP anywhere. The `#pragma omp`
      loops then run single-threaded; results are unchanged.
    """
    if os.environ.get("NEMOTRON_OCR_NO_OPENMP", "0") == "1":
        return [], []
    if sys.platform == "darwin":
        prefix = os.environ.get("NEMOTRON_OCR_LIBOMP_PREFIX", "")
        if not prefix and shutil.which("brew"):
            import subprocess
            try:
                prefix = subprocess.run(
                    ["brew", "--prefix", "libomp"], capture_output=True, text=True, check=True
                ).stdout.strip()
            except Exception:
                prefix = ""
        if not prefix:
            for cand in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
                if Path(cand).exists():
                    prefix = cand
                    break
        if not prefix or not (Path(prefix) / "include" / "omp.h").exists():
            print(
                "WARNING: libomp not found (brew install libomp, or set NEMOTRON_OCR_LIBOMP_PREFIX). "
                "Building without OpenMP."
            )
            return [], []
        return (
            ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
            [f"-L{prefix}/lib", "-lomp", f"-Wl,-rpath,{prefix}/lib"],
        )
    if os.name == "nt":
        return ["/openmp"], []
    return ["-fopenmp"], ["-fopenmp"]


WITH_CUDA = _cuda_requested()

includes = [f"-I{os.getcwd()}/cpp/trove"]

libs = []

common_args = ["-std=c++17"]
omp_compile, omp_link = _openmp_flags()
gcc_args = omp_compile + (["-DOMP_NESTED=true"] if omp_compile else [])
cuda_args = []
link_args = list(omp_link)

def _host_has_avx2() -> bool:
    """True if the build machine's CPU has AVX2 (unknown counts as yes, as upstream)."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                return " avx2" in f.read()
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.leaf7_features"],
                                 capture_output=True, text=True).stdout
            return "AVX2" in out.upper()
    except OSError:
        pass
    return True


_target_arch = os.environ.get("ARCH", "").lower() or platform.machine().lower()
# -mavx2 only where the CPU has it (pre-Haswell Xeons such as E5 v1/v2 don't).
# NEMOTRON_OCR_AVX2=0/1 overrides, e.g. when building for a different machine.
_avx2_env = os.environ.get("NEMOTRON_OCR_AVX2")
_use_avx2 = _avx2_env != "0" if _avx2_env is not None else _host_has_avx2()
if _target_arch in ("x86_64", "amd64") and _use_avx2:
    gcc_args += ["-mavx2"]
elif _target_arch in ("x86_64", "amd64"):
    print("build-extension: building without -mavx2 (CPU has no AVX2)")

files = sorted(glob("cpp/**/*.cpp", recursive=True))
if WITH_CUDA:
    files = sorted(files + glob("cpp/**/*.cu", recursive=True))
files = [f for f in files if "trove/tests" not in f]

# debug=True
debug = False
if debug:
    common_args += ["-g", "-O0"]
    cuda_args += ["-G"]
else:
    common_args += ["-O3", "-DNDEBUG"]

if WITH_CUDA:
    # Decide architectures in this order of precedence:
    # 1) Respect TORCH_CUDA_ARCH_LIST if provided
    # 2) Detect from the visible GPU at runtime
    # 3) Fallback to a safe default list
    arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    cuda_architectures = _parse_arch_list(arch_env) if arch_env else _detect_arch_list()

    from torch.utils.cpp_extension import CUDA_HOME  # type: ignore

    cuda_includes = [f"-I{Path(CUDA_HOME) / 'include'}"] + includes
    common_args += ["-DWITH_CUDA"]
    compile_args = {
        "cxx": common_args + gcc_args + cuda_includes,
        "nvcc": common_args + cuda_args + cuda_includes + cuda_architectures,
    }
else:
    compile_args = {"cxx": common_args + gcc_args + includes}


def build() -> None:
    if os.environ.get("BUILD_CPP_EXTENSION", "1") != "1":
        print("Environment variable BUILD_CPP_EXTENSION=1 not set. Skipping build.")
        sys.exit(0)

    from setuptools import Distribution
    from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

    print(f"Building _nemotron_ocr_cpp ({'CUDA' if WITH_CUDA else 'CPU-only'}, {len(files)} sources)")
    ext_cls = CUDAExtension if WITH_CUDA else CppExtension
    ext_modules = [
        ext_cls(
            "_nemotron_ocr_cpp",
            files,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            libraries=libs,
        )
    ]

    distribution = Distribution({"name": "nemotron_ocr_cpp", "ext_modules": ext_modules})

    build_ext = BuildExtension.with_options(parallel=False)
    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = Path("src/nemotron_ocr_cpp") / output.relative_to(cmd.build_lib)

        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build()
