# Progress

Running log for the portable ONNX port of Nemotron OCR v2. Newest decisions at the
bottom of each section.

## Status

| Milestone | State | Commit |
|---|---|---|
| Import nemotron-ocr 2.0.2 sdist (unmodified) | done | `9850078` |
| M0: pipeline map (`docs/ARCHITECTURE.md`) | done | `622cdf9` |
| M1: CPU-only build | done | see git log |
| M2: golden reference outputs | not started | |
| M3: ONNX export | not started | |
| M4: ONNX Runtime pipeline | not started | |
| M5: validation harness + `docs/TARGETS.md` | not started | |

## Environment used here

Ubuntu 24.04 x86-64, 4 vCPU, 15 GB RAM, no GPU. Python 3.12, torch 2.11.0+cpu,
onnx 1.23.2, onnxruntime 1.31.1 (CPU). The sdist was fetched from
files.pythonhosted.org and checked against its PyPI sha256; weights come from
`nvidia/nemotron-ocr-v2` via `scripts/fetch_weights.sh`.

## M1: what changed

- `scripts/build-extension.py`: builds a `CppExtension` from `cpp/**/*.cpp` only when
  CUDA isn't usable (no nvcc under `CUDA_HOME`, a CPU-only torch, or
  `NEMOTRON_OCR_CPU_ONLY=1`). `-DWITH_CUDA` and the CUDA include path (taken from
  `CUDA_HOME`, no longer hard-coded `/usr/local/cuda`) are added only for CUDA builds.
  `-mavx2` only on x86-64. OpenMP flags per platform (see macOS notes).
- C++: every call into CUDA code is behind `#ifdef WITH_CUDA`
  (`quad_rectify.h`, `grid_sample.h`, `geometry_api.cpp`, `non_maximal_suppression.cpp`,
  `scope_timer.h`). In a CPU build, passing CUDA tensors raises
  "built without CUDA support". The CUDA-only distance op isn't registered in CPU
  builds; the module exposes `WITH_CUDA` (bool).
- NMS allocated its outputs in pinned memory unconditionally, which throws without a
  CUDA driver. It now pins only when `at::hasCUDA()`.
- `NemotronOCR` / `NemotronOCRV2` take `device=` (default: cuda if available, else
  cpu) and `precision=` (`"fp16"` = upstream autocast behaviour, default on CUDA;
  `"fp32"` = no autocast and no half conversions, default elsewhere). TF32 is only
  enabled in fp16 mode. `verbose_post` timing syncs only on CUDA.
- `torch.load` uses `map_location="cpu"` for all checkpoints.
- The detector no longer downloads ImageNet RegNet weights from torch hub at
  construction (see Decisions).
- `models/quad_distance.py`: pure-PyTorch `ragged_quad_all_2_all_distance_v2`, used
  whenever tensors aren't on CUDA or the extension has no CUDA. Tested in
  `tests/test_quad_distance.py` on hand-computed quads, and against a step-by-step
  port of the kernel on random quads.
- `example.py` (from the Hugging Face repo, Apache-2.0) with `--device` and
  `--precision`.

Acceptance: `python example.py weights/ocr-example-input-1.png --model-dir
weights/v2_english` (and `v2_multilingual`) runs on CPU in FP32 and prints the
expected bank-reconciliation text (24 / 27 regions at paragraph level). About 2.4 s
per page at 1024 px on this 4-vCPU machine, ~90% of it in the detector.

### macOS (not tested here)

Apple clang has no built-in OpenMP. The build script looks for Homebrew `libomp`
(`brew install libomp`; prefix from `brew --prefix libomp`, `/opt/homebrew/opt/libomp`,
`/usr/local/opt/libomp`, or `NEMOTRON_OCR_LIBOMP_PREFIX`) and uses
`-Xpreprocessor -fopenmp` with `-lomp` and an rpath. If none is found it warns and
builds without OpenMP. `NEMOTRON_OCR_NO_OPENMP=1` forces that anywhere. Without
OpenMP the `#pragma omp` loops (grid sample, NMS) run single-threaded; results are
identical.

### ARM CPUs (not tested here)

Kept compatible by construction; the owner will test on an ARM device.

- The C++ sources use no x86 intrinsics (checked: no `immintrin.h`, `_mm*`, `__m256`
  etc. outside the unused trove CUDA library). `-mavx2` was the only x86-specific flag
  and is now added only on x86-64 (`platform.machine()` or `ARCH`).
- aarch64 Linux: GCC with `-fopenmp`, same as x86. PyTorch CPU and ONNX Runtime
  publish aarch64 wheels.
- Apple Silicon: see macOS above.
- Windows on ARM: MSVC `/openmp`; ONNX Runtime has ARM64 wheels. PyTorch support on
  Windows ARM64 is newer and may limit the glue ops there (see M4 note on libtorch).

## Decisions

1. **Upstream import goes on top of the owner's "Initial commit"** (README only), so
   the sdist is the second commit, not the first. Every later change is still a
   diff against it.
2. **Default precision.** FP32 is the default off-CUDA. On CUDA the PyTorch
   reference keeps upstream's fp16 autocast by default (`precision="fp32"` for the
   P40). The ONNX export and runtime are FP32 only.
3. **ImageNet download.** `regnet_x_8gf(pretrained=True)` both selects the ImageNet
   block widths (which the checkpoint needs) and downloads the ImageNet weights
   (which `detector.pth` overwrites). The download is now skipped with a context
   manager in `regnet.py`; architecture and numbers are unchanged.
4. **Distance op implementation.** The pipeline and export use the symmetric
   vertex-to-edge form (exact in real arithmetic for x/y factors = 1, the only values
   upstream passes; ~1 ulp float differences), which needs 4× less memory than a
   literal port of the kernel ([N, N, 4, 4] vs [N, N, 4, 4, 4, 2]). The literal port
   is kept as `quad_all_2_all_distance_reference` for tests and is used for other
   factors.

## Open questions for the owner

1. **Relational null-column collision (upstream behaviour).** For pages with ≤ 15
   regions, the relational model's `scatter` writes several values to the null
   column, and on CPU the last one (a padded slot) wins; on CUDA the winner is
   unordered. Details in `docs/ARCHITECTURE.md` §5. The port reproduces the CPU
   behaviour exactly and doesn't fix it, because fixing it changes model output.
   Say if you'd rather have it fixed (opt-in) later.
2. The repository description on GitHub still says "nvidia-ocr-v2"; optional rename.
