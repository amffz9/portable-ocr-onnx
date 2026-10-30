# Licences, notices and credits

This repository combines code under several licences with model weights under a
separate model licence. This file lists what is used, where it comes from and what
it requires. It is a summary, not legal advice; the licence texts govern.

## Summary

| What | Where | Licence | Copyright / origin |
|---|---|---|---|
| Upstream `nemotron-ocr` 2.0.2 code (Python, C++) | `src/nemotron_ocr/`, `src/nemotron_ocr_cpp/`, `cpp/`, `scripts/build-extension.py`, `hatch_build.py`, `pyproject.toml` | Apache-2.0 ([`LICENSE`](LICENSE)) | © 2025 NVIDIA CORPORATION & AFFILIATES |
| `example.py` | repo root | Apache-2.0 | © 2025 NVIDIA CORPORATION & AFFILIATES, from the `nvidia/nemotron-ocr-v2` Hugging Face repo |
| trove (CUDA library) | `cpp/trove/` | BSD-3-Clause ([`cpp/trove/LICENSE`](cpp/trove/LICENSE)) | © 2013 NVIDIA Corporation |
| Clipper 6.4.0 (polygon clipping) | `cpp/third_party/clipper/` | Boost Software License 1.0 ([`LICENSES/BSL-1.0.txt`](LICENSES/BSL-1.0.txt), also next to the source) | © 2010–2015 Angus Johnson |
| Changes to upstream files and all new code (`src/nemotron_ocr_onnx/`, `scripts/*.py` other than the build script, `tests/`, `docs/`) | throughout | Apache-2.0 ([`LICENSE`](LICENSE)) | © 2026 the portable-ocr-onnx contributors |
| Nemotron OCR v2 model weights | downloaded to `weights/` (not in git) | NVIDIA Open Model License ([`LICENSES/NVIDIA-Open-Model-License.txt`](LICENSES/NVIDIA-Open-Model-License.txt)); notice in [`NOTICE`](NOTICE) | NVIDIA Corporation, [`nvidia/nemotron-ocr-v2`](https://huggingface.co/nvidia/nemotron-ocr-v2) |
| Exported ONNX graphs | written to `onnx/` (not in git) | Same as the weights (a Derivative Model) | as above |
| Synthetic test pages | `tests/images/` | Apache-2.0, same as the code | original text and layouts written for this repository |
| Golden reference outputs | `tests/golden/` | Apache-2.0, same as the code | outputs of the model on the test pages; `sample.*` also derive from NVIDIA's `ocr-example-input-1.png` |

## The model weights and exported graphs

The weights are governed by the **NVIDIA Open Model License Agreement** (version
of 24 October 2025): <https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/>.
A copy is in `LICENSES/NVIDIA-Open-Model-License.txt`. The model card also points
to Apache-2.0 as "additional information". Weights are never committed to this
repository; `scripts/fetch_weights.sh` downloads them from Hugging Face, so each
user obtains them under that agreement directly.

In short, the agreement:

- allows commercial use, and creating and distributing **Derivative Models**
  ("modifications to the Model, works based on the Model, and any other derivative
  works"). The ONNX graphs written by `scripts/export_onnx.py` are a Derivative
  Model.
- requires that anyone who **distributes the model or a Derivative Model** gives
  recipients a copy of the agreement and includes this notice in a "Notice" text
  file: "Licensed by NVIDIA Corporation under the NVIDIA Open Model License".
  [`NOTICE`](NOTICE) contains exactly that line. `scripts/export_onnx.py` copies
  `NOTICE` and the agreement (`LICENSE.txt`) into every `onnx/<variant>/` directory,
  so that directory can be shared as it is. Keep both files with the graphs.
- claims no ownership of **outputs** (OCR text, boxes); you are responsible for
  them and their use.
- requires use consistent with NVIDIA's Trustworthy AI terms
  (<https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/>), and terminates
  if a guardrail in the model is circumvented. (This port changes the file format
  and runtime only, not the model's weights or behaviour.)
- grants no right to NVIDIA's trademarks beyond describing the model's origin. This
  project therefore isn't named after NVIDIA or Nemotron, and isn't affiliated with
  or endorsed by NVIDIA.

## Apache-2.0 obligations for the code

- The full licence is in [`LICENSE`](LICENSE).
- Every upstream file changed here carries a "Modified from nemotron-ocr 2.0.2
  (NVIDIA)" line at the top describing the change (Apache-2.0 §4(b)).
- Upstream copyright headers (`SPDX-FileCopyrightText`) are kept unchanged.
- The upstream sdist (`nemotron-ocr` 2.0.2 on PyPI, sha256
  `703ae32bf7172da8985fca90ff22584fd9622ee04ce877d0cc6d2db4be7dce5b`) is imported unmodified as the second commit, so every change is
  visible in git history.

## Runtime dependencies (installed separately, not bundled)

| Package | Licence |
|---|---|
| PyTorch, torchvision | BSD-3-Clause |
| ONNX Runtime (`onnxruntime`, `-gpu`, `-directml`) | MIT |
| ONNX (export only) | Apache-2.0 |
| NumPy | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) |
| Shapely | BSD-3-Clause |
| huggingface_hub | Apache-2.0 |
| hatchling, editables (build) | MIT |
| ninja (build) | Apache-2.0 |
| Poppler `pdftoppm` (optional, PDF input in `scripts/ocr.py`, run as a separate program) | GPL-2.0-or-later |

GPU builds of ONNX Runtime pull in NVIDIA CUDA / cuDNN runtime libraries, which are
under NVIDIA's own licence terms for those packages.

## Fonts used to render the test pages

The PNGs in `tests/images/` were rendered by `scripts/make_test_pages.py` with these
fonts from Ubuntu 24.04 packages. The font files are not included in this
repository; the rendered images are not font software. Credited with thanks:

| Font | Package | Licence |
|---|---|---|
| DejaVu Serif / Sans / Sans Mono | `fonts-dejavu-core` 2.37 | Bitstream Vera Fonts licence (DejaVu changes public domain) |
| Liberation Sans | `fonts-liberation` 2.1.5 | SIL Open Font License 1.1 |
| GNU FreeFont (FreeSerif Italic) | `fonts-freefont-ttf` 20211204 | GPL-3.0-or-later with font exception |
| WenQuanYi Zen Hei | `fonts-wqy-zenhei` 0.9.45 | GPL-2.0 with font embedding exception |

## Credits

- **NVIDIA** for Nemotron OCR v2: the model, its weights and the `nemotron-ocr`
  inference package this work is built on
  ([model card](https://huggingface.co/nvidia/nemotron-ocr-v2),
  [PyPI](https://pypi.org/project/nemotron-ocr/)).
- **Angus Johnson** for the Clipper polygon clipping library.
- The **ONNX** and **ONNX Runtime** projects, and **PyTorch**, whose exporter
  produced the graphs.
- The DejaVu, Liberation, GNU FreeFont and WenQuanYi font projects.
