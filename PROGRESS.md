# Progress

Running log for the portable ONNX port of Nemotron OCR v2. Newest decisions at the
bottom of each section.

## Status

| Milestone | State | Commit |
|---|---|---|
| Import nemotron-ocr 2.0.2 sdist (unmodified) | done | `9850078` |
| M0: pipeline map (`docs/ARCHITECTURE.md`) | done | `622cdf9` |
| M1: CPU-only build | done | see git log |
| M2: golden reference outputs | done | `36ba73f` |
| M3: ONNX export | done | see git log |
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

## M2: golden outputs

- `scripts/make_test_pages.py` renders 11 synthetic pages (`tests/images/`, 484 KB,
  committed): letter, two-column, receipt, form, rotated text, sparse (< 16 regions),
  slide, low contrast, dense, multilingual (Latin/Cyrillic/Greek/CJK/Hangul), blank.
  Plus the upstream sample `weights/ocr-example-input-1.png` ("sample").
- `scripts/make_golden.py` runs `NemotronOCRV2` (CPU, FP32) and hooks every network:
  detector in/out, NMS quads, sampler grids and outputs, recognizer in/out,
  relational matrices, and final predictions at word / sentence / paragraph level
  (with pixel boxes).
- Two tiers: `tests/golden/` (committed, ~9 MB; small tensors whole, large ones
  strided, stride recorded in the key as `__s…`) and `tests/golden_full/`
  (`--full`, ~2.1 GB, gitignored; what the export check uses). Regenerate the full
  set locally with `scripts/make_golden.py --full`.

## M3: ONNX export

`scripts/export_onnx.py --variant {english,multilingual}` writes
`onnx/<variant>/{detector,sampler,recognizer,relational}.onnx` (opset 17, FP32,
legacy TorchScript exporter) plus `model_meta.json`, `charset.txt` and
`model_config.json`, then checks each graph with `onnx.checker` and ONNX Runtime
CPU against `tests/golden_full`, writing `export_report.json`. Each downstream graph
is fed the golden inputs, so every graph is checked in isolation.

The trace wrappers are in `src/nemotron_ocr_onnx/export_modules.py`. They reuse the
upstream modules and weights unchanged; only control flow the tracer can't follow is
restated:

- **sampler**: the C++ `quad_rectify_forward` and `indirect_grid_sample_forward`
  rewritten as tensor ops. The C++ sampler (align-corners, clamped neighbours,
  0 outside [-1, 1]) equals `GridSample(bilinear, border, align_corners=1)` times an
  in-range mask. One image per call (`image_index`), so no custom gather op.
- **recognizer**: also outputs `token_ids` (argmax) and `token_probs` (max softmax),
  which the pipeline needs, so the runtime need not fetch the multilingual logits
  ([n, 128, 14247]).
- **relational**: one image per call. `topk(min(n-1, 15))` becomes a fixed
  `topk(15)` over distances padded with +inf, with masked slots zeroed as upstream;
  the scatter uses a spare column for masked slots and reproduces upstream's CPU
  null-column value (open question 1). The pure-torch distance op is used (bool
  `EyeLike` replaced by an arange comparison, since ORT doesn't implement it).
- The PyTorch MHA fast path (`_transformer_encoder_layer_fwd`) is turned off during
  export so the transformer traces as standard ops.

All four graphs use only the default ONNX domain (no custom ops). Sizes: detector
181 MB, sampler < 0.1 MB, recognizer 25 MB (english) / 145 MB (multilingual),
relational 9 MB.

`tests/test_onnx_export.py` covers shapes the goldens don't: relational with n = 1, 2,
15, 16, 17; sampler on the second image of a batch with non-square features;
detector at batch 2, 512×768.

### Results vs goldens (ORT 1.31.1 CPU, all 12 golden images per variant)

| output | english max abs | multilingual max abs | largest ref value |
|---|---|---|---|
| detector.conf | 3.6e-4 | 7.6e-4 | 8.5 / 4.2 |
| detector.rboxes | 1.09 | 5.1e-2 | 3.2e5 / 1.4e3 |
| detector.features | 2.6e-4 | 6.5e-5 | 73 / 32 |
| sampler.rec_input | 1.8e-7 | 4.8e-7 | 1.9 / 2.2 |
| sampler.rel_input | 1.2e-7 | 2.4e-7 | 1.1 / 2.0 |
| recognizer.logits | **4.5e-5** | **9.1e-5** | 11 / 22 |
| recognizer.features | 1.2e-5 | 7.6e-6 | 2.9 / 3.1 |
| recognizer.token_probs | 7.6e-6 | 1.2e-5 | 1.0 |
| relational.words | 2.1e-4 | 1.3e-4 | 69 / 132 |
| relational.lines | 3.6e-4 | 6.3e-5 | 57 / 35 |
| relational.line_log_var | 4.3e-5 | 2.8e-5 | 11 / 17 |

Token ids: 0 / 30464 (english) and 0 / 23680 (multilingual) mismatches. Minimum
cosine similarity for every output ≥ 0.99999999975.

Recognizer logits meet the 1e-4 target. The outputs above 1e-4 in absolute terms
are all ~1e-5 to 1e-6 relative to their magnitude, which is FP32 accumulation-order
noise rather than a math difference:

- **detector**: a RegNet-X-8GF backbone plus FPN; ORT and PyTorch use different
  convolution algorithms (summation order) for ~100 layers, so differences grow
  with depth. `rboxes` are the exponentiated distance outputs; the 1.09 is at a
  value of ~3e5 (background pixels far below the confidence threshold), i.e. 3e-6
  relative. Downstream it is invisible: NMS quads are computed from the golden
  pipeline and the M4 end-to-end check compares final boxes in pixels.
- **relational**: logits of up to ~130, after a transformer with softmax attention;
  2e-4 at that scale is ~1.5e-6 relative. These feed a thresholded graph
  clustering, so M4 compares the final grouping.

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
