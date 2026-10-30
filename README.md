# portable-ocr-onnx

Run **NVIDIA Nemotron OCR v2** anywhere ONNX Runtime runs: CPU (x86-64 and ARM),
older NVIDIA GPUs such as the Tesla P40, AMD GPUs, Apple Silicon and Windows
DirectML. No CUDA toolkit is needed.

Nemotron OCR v2 is a strong document OCR model: it finds text, reads it and groups
it into lines and paragraphs in reading order. Upstream it runs only on recent
NVIDIA GPUs with CUDA. This project exports its networks to standard ONNX, builds
its C++ helpers without CUDA, and checks that the result gives **the same output as
the original**.

- Same text as the original PyTorch model on every test page, with box positions
  within 0.001 px.
- About 2 s per page on a 4-core CPU, faster than the original PyTorch model on
  the same CPU.
- English and multilingual variants. Multilingual covers Latin scripts, Cyrillic,
  Greek, Chinese, Japanese and Korean.
- A command-line tool for images and PDFs, and a Python class with the same API as
  upstream.

> Not affiliated with or endorsed by NVIDIA. The model weights are NVIDIA's and are
> downloaded from Hugging Face under the NVIDIA Open Model License; see
> [Licences](#licences-and-credits).

## Contents

- [Status](#status)
- [Install](#install)
- [Command-line use](#command-line-use)
- [Python use](#python-use)
- [Choosing a variant and settings](#choosing-a-variant-and-settings)
- [Accuracy and speed](#accuracy-and-speed)
- [GPUs and other platforms](#gpus-and-other-platforms)
- [How it works](#how-it-works)
- [Development](#development)
- [Licences and credits](#licences-and-credits)

## Status

| Platform | ONNX Runtime provider | State |
|---|---|---|
| Linux x86-64 CPU | CPU | **validated** (all checks pass, both variants) |
| Linux aarch64 / Apple Silicon / Windows CPU | CPU | supported by design, not yet validated |
| NVIDIA GPUs incl. Tesla P40 (Pascal) | CUDA | steps documented, not yet validated |
| AMD GPUs | MIGraphX | steps documented, not yet validated |
| macOS | CoreML | steps documented, not yet validated |
| Windows, any DirectX 12 GPU | DirectML | steps documented, not yet validated |

"Validated" means `scripts/validate.py` passes on that machine. If you run it on a
platform that isn't validated yet, please share the JSON report it writes.

## Install

You need Python 3.11–3.13, `git`, a C++ compiler (`g++` on Linux; Xcode command-line
tools on macOS; Visual Studio Build Tools on Windows), and about 3 GB of free disk
space. On macOS, `brew install libomp` is optional and enables multithreading in the
C++ helpers.

### 1. Clone and create an environment

```bash
git clone https://github.com/amffz9/portable-ocr-onnx.git
cd portable-ocr-onnx
python3.12 -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install the dependencies and the package

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow "shapely>=2.1.2,<3" huggingface_hub setuptools ninja hatchling editables onnx onnxruntime
pip install --no-build-isolation -e .
```

- **PyTorch is the CPU build on purpose.** It only runs the small pre- and
  post-processing steps; the networks run in ONNX Runtime.
- **macOS:** drop `--index-url ...` from the first line (PyPI's macOS wheels are
  already CPU-only).
- **GPU:** install a different `onnxruntime` package instead of `onnxruntime`; see
  [GPUs and other platforms](#gpus-and-other-platforms).
- The last command compiles the C++ helpers, which takes a minute or two.

### 3. Download the weights and export the ONNX models

```bash
scripts/fetch_weights.sh all                                     # or: english | multilingual
python scripts/export_onnx.py --variant english --no-check       # writes onnx/english/
python scripts/export_onnx.py --variant multilingual --no-check  # writes onnx/multilingual/
```

- **Weights:** `fetch_weights.sh` downloads them from
  [`nvidia/nemotron-ocr-v2`](https://huggingface.co/nvidia/nemotron-ocr-v2) into
  `weights/`. Without bash (e.g. Windows), run
  `hf download nvidia/nemotron-ocr-v2 --local-dir weights` instead.
- **Export:** takes a few minutes and about 3 GB of RAM, and gives the same result
  on any OS.
- **Sizes:** `onnx/english/` is about 210 MB; `onnx/multilingual/` is about 320 MB.
- **Copying:** the ONNX files are portable. You can export once and copy the `onnx/`
  folder to other machines, which still need steps 1–2. Keep the `NOTICE` and
  `LICENSE.txt` files that sit next to the graphs.

### 4. Check the installation

```bash
python scripts/validate.py
```

This runs both variants over the test pages and compares every result with the
original PyTorch model's output stored in `tests/golden/`. It prints the platform,
the ONNX Runtime provider each network actually used, the text match rate, box
error, pages per second and peak memory. It exits with an error on any mismatch.
Expected result: `PASS`, with 100% text match.

## Command-line use

`scripts/ocr.py` reads images (PNG, JPEG, …) and PDFs. PDFs are rendered with
Poppler's `pdftoppm` (`apt install poppler-utils`, `brew install poppler`).

```bash
python scripts/ocr.py page.png                       # print the text, paragraph by paragraph
python scripts/ocr.py scan.pdf                       # every page of a PDF
python scripts/ocr.py *.jpg --level sentence         # one line per text line
python scripts/ocr.py scan.pdf --json out.json       # text + confidence + pixel boxes as JSON
python scripts/ocr.py scan.pdf --min-confidence 0.9  # drop low-confidence regions (gibberish)
python scripts/ocr.py doc.pdf --variant multilingual # non-Latin scripts
python scripts/ocr.py doc.pdf --providers cuda -v    # pick an accelerator, log what was used
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--variant` | `english` | `english` or `multilingual` (see below) |
| `--level` | `paragraph` | `word`, `sentence` (single lines) or `paragraph` (blocks in reading order) |
| `--min-confidence` | `0` | skip regions scoring below this (0–1) |
| `--providers` | `auto` | `auto`, or one or more of `cpu`, `cuda`, `migraphx`, `coreml`, `dml` |
| `--threads` | ORT default | CPU threads |
| `--dpi` | `150` | PDF render resolution (more doesn't help: pages are resized to 1024 px) |
| `--json FILE` | — | write results to a JSON file instead of printing |
| `--model-dir` | `onnx/<variant>` | use ONNX files from another folder |

JSON output looks like this:

```json
{"variant": "english", "level": "paragraph",
 "providers": {"detector": "CPUExecutionProvider", "...": "..."},
 "pages": [{"page": "ocr-example-input-1.png", "width": 889, "height": 761,
            "regions": [{"text": "Trunch Parish Council", "confidence": 0.9383,
                         "box_px": [298.0, 14.6, 488.8, 34.0]},
                        "..."]}]}
```

`box_px` is `[x0, y0, x1, y1]` in pixels of the rendered page, with y running
downwards.

## Python use

```python
from nemotron_ocr_onnx import NemotronOCRONNX

ocr = NemotronOCRONNX("onnx/english", providers="auto")   # create once, reuse

for region in ocr("page.png", merge_level="paragraph"):
    print(f"{region['confidence']:.2f}  {region['text']}")

pages = ocr(["p1.png", "p2.png"], merge_level="sentence")  # a list in, a list of lists out
```

**Inputs:** a file path, an `H×W×3` uint8 NumPy array (RGB), a torch tensor, or a
list of these.

**Output:** for each image, a list of regions in reading order. Each region is a
dict:

| Key | Meaning |
|---|---|
| `text` | recognised text |
| `confidence` | 0–1. Clean printed text typically scores 0.93+; handwriting and garbled map labels mostly score below 0.9 |
| `left`, `right` | x of the box edges, as a fraction of the image width |
| `lower`, `upper` | y of the box edges, as a fraction of the image height. **`lower` is the smaller y (top of the box) and `upper` the larger y (bottom).** These names come from upstream. |

Pixel box: `(left*W, lower*H, right*W, upper*H)`.

**Constructor options:**

| Argument | Default | Meaning |
|---|---|---|
| `model_dir` | — | `onnx/english` or `onnx/multilingual` |
| `providers` | `"auto"` | `"auto"` tries CUDA → MIGraphX → CoreML → DirectML → CPU. Or a name/alias (`"cpu"`, `"cuda"`), or a list such as `[("CUDAExecutionProvider", {"device_id": 1})]`. Naming a provider your ONNX Runtime doesn't have raises an error. |
| `intra_op_threads` | `0` (ORT default) | CPU threads |
| `detector_max_batch_size` | 1 on CPU, 8 on GPU | images per detector run |
| `recognizer_chunk_size` | `32` | text regions per recognizer run; lower it to save memory |
| `skip_relational` | `False` | skip line/paragraph grouping (faster; word-level results only) |
| `detector_only` | `False` | return boxes without reading text |
| `infer_length` | `1024` | size pages are resized to before detection (see below) |

`ocr.session_providers` shows which provider each of the four networks actually
got. It is also logged at INFO level. If a GPU provider fails to start, that
network falls back to CPU with a warning.

**Merge levels:**

- `word`: single words (the multilingual variant detects lines, so its "words"
  are often whole lines).
- `sentence`: single text lines. Best for filtering by confidence.
- `paragraph`: lines grouped into blocks in reading order. Best for readable text.

## Choosing a variant and settings

- **English documents: use `english`.** It is the most accurate variant for Latin
  text, including rotated or skewed text and noisy scans.
- **Other scripts: use `multilingual`.** It covers Chinese, Japanese, Korean,
  Cyrillic, Greek and accented Latin. It reads whole lines at once, and **very long
  lines of small text get garbled**. Don't use it for dense English pages.
- **Filtering gibberish:** use `merge_level="sentence"` and drop lines below about
  0.9 confidence (`--min-confidence 0.9`). On a real scanned document this removed
  the signatures and garbled map labels and kept every typed line. Tune the cutoff
  on your own documents.
- **Very small, dense text:** every page is resized so its longer side is 1024 px.
  Small text loses detail in that resize. On a dense page of 12 px text the
  English variant scored 90, where Tesseract and PaddleOCR scored 99–100. A larger `infer_length` (e.g. 1536) is possible but
  slower and not validated.
- **Maps, drawings and photos:** no OCR engine reads these well. Detect such pages
  and describe them with a vision model instead.
- **PDFs that already contain text** (exported from Word and similar): extract the
  text layer (`pdftotext`) instead of running OCR.

## Accuracy and speed

Compared with other open OCR engines on the same CPU (4 vCPU, x86-64). Score =
word F1: the share of words read correctly, ignoring order (100 = perfect).

| Test set | Nemotron EN | Nemotron multilingual | Tesseract 5.3 | PaddleOCR 3.7 (PP-OCRv6) |
|---|---|---|---|---|
| clean pages (10) | **98.6** | 88.5 | 93.5 | 91.2 |
| degraded scans (10: skew, blur, noise, JPEG) | **98.2** | 82.8 | 81.8 | 90.7 |
| real scanned municipal document | 97.4 | **99.0** | 95.1 | 93.0 |
| multilingual page (Latin, Cyrillic, Greek, CJK, Hangul) | 31.4 | **96.2** | 28.6 | 73.9 (English model) |
| seconds per page | 1.7–2.1 | 1.8–3.0 | **0.4–0.7** | 10–28 |

- **Strengths:** rotated and slanted text (100 vs 21–48 for the others) and noisy
  scans.
- **Weaknesses:** very small, dense text (90 vs 99–100 for the others), and the
  multilingual variant on long lines.
- **Caveats:** the test pages are synthetic plus one real document, so treat these
  as indicative. Tesseract and PaddleOCR ran with default settings, and PaddleOCR
  ran with its CPU acceleration (oneDNN) off because of a bug in paddlepaddle 3.3.

**Against the original PyTorch model** on the same CPU, the output is identical and
the speed is about 2.4 s → 1.7 s per page for English and 7–9 s → 1.9 s for
multilingual. Peak memory is about 3 GB.

## GPUs and other platforms

The networks are FP32 and use only standard ONNX operators (opset 17), so any ONNX
Runtime build can run them. To use a GPU, swap the `onnxruntime` package:

| Hardware | Install instead of `onnxruntime` | Provider |
|---|---|---|
| NVIDIA (incl. Tesla P40) | `onnxruntime-gpu[cuda,cudnn]` (**< 1.27 for Pascal cards**) | `cuda` |
| AMD | AMD's MIGraphX build of ONNX Runtime | `migraphx` |
| Apple | `onnxruntime` (CoreML is included on macOS) | `coreml` |
| Windows, any DX12 GPU | `onnxruntime-directml` | `dml` |

Exact commands, version pins and troubleshooting for each are in
[docs/TARGETS.md](docs/TARGETS.md). The pre- and post-processing always run on the
CPU, so PyTorch stays the CPU build on every platform.

## How it works

The pipeline has four ONNX networks plus CPU glue code:

```
image ─► preprocess (pad, resize to 1024)                                    [CPU, torch]
      ─► detector.onnx    text heat-map, rotated boxes, feature map          [ONNX Runtime]
      ─► prefilter + NMS  → text-region quadrilaterals                       [CPU, C++]
      ─► sampler.onnx     crops each region out of the feature map           [ONNX Runtime]
      ─► recognizer.onnx  reads the characters of each region                [ONNX Runtime]
      ─► relational.onnx  links regions into lines and paragraphs            [ONNX Runtime]
      ─► decode           text, reading order, boxes in original pixels      [CPU, C++/Python]
```

- **Unchanged from upstream:** the network weights and math; the export only
  changes the file format.
- **Rewritten in standard ONNX operators:** the parts of upstream that were custom
  CUDA code (region cropping, neighbour search).
- **Edge cases matched exactly:** e.g. how pages with fewer than 16 text regions
  are handled.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) maps the upstream pipeline;
[PROGRESS.md](PROGRESS.md) records every decision, the numeric differences per
network and the remaining open questions.

## Development

```bash
python -m pytest tests/                                # unit + ONNX tests (need weights and onnx/)
python scripts/validate.py --variants english          # end-to-end check against tests/golden
python scripts/export_onnx.py --variant english        # re-export and compare every network's tensors
python scripts/make_golden.py --full                   # regenerate goldens (incl. 2 GB full tensors)
python scripts/make_test_pages.py                      # re-render the synthetic test pages
```

The original PyTorch pipeline still works (on CPU in FP32, or on CUDA as upstream):

```bash
python example.py weights/ocr-example-input-1.png --model-dir weights/v2_english --device cpu
```

Build options (environment variables for `pip install -e .`):

- `NEMOTRON_OCR_CPU_ONLY=1`: skip CUDA even if `nvcc` is present.
- `NEMOTRON_OCR_NO_OPENMP=1`: build without OpenMP.
- `NEMOTRON_OCR_LIBOMP_PREFIX`: libomp location on macOS.
- `BUILD_CPP_FORCE=1`: force a rebuild.

### Repository layout

| Path | Contents |
|---|---|
| `src/nemotron_ocr_onnx/` | `NemotronOCRONNX`, provider selection, ONNX export wrappers |
| `src/nemotron_ocr/`, `src/nemotron_ocr_cpp/`, `cpp/` | upstream `nemotron-ocr` 2.0.2, modified to build without CUDA (each changed file says so at the top) |
| `scripts/ocr.py` | command-line OCR for images and PDFs |
| `scripts/validate.py` | end-to-end validation on any machine |
| `scripts/export_onnx.py` | ONNX export and per-network checks |
| `scripts/fetch_weights.sh` | weight download from Hugging Face |
| `scripts/make_test_pages.py`, `scripts/make_golden.py` | test pages and reference outputs |
| `tests/` | unit tests, test pages (`images/`), reference outputs (`golden/`) |
| `docs/TARGETS.md` | per-platform install steps |
| `docs/ARCHITECTURE.md` | how the upstream pipeline works |
| `PROGRESS.md` | development log, results, open questions |

## Licences and credits

- **Code:** Apache 2.0 ([LICENSE](LICENSE)). Most of it comes from NVIDIA's
  `nemotron-ocr` 2.0.2; changes and new code are by the portable-ocr-onnx
  contributors.
- **Bundled libraries:** trove (BSD-3-Clause) and Clipper (Boost Software
  License).
- **Model weights:** NVIDIA Open Model License
  ([copy](LICENSES/NVIDIA-Open-Model-License.txt), [NOTICE](NOTICE)). They are
  downloaded from Hugging Face, never stored in this repository. Commercial use is
  allowed. NVIDIA claims no ownership of OCR output.
- **Exported ONNX files:** count as a "Derivative Model" of the weights. If you
  share them, include the `NOTICE` and `LICENSE.txt` files that the export puts
  next to them.

Full details, dependency licences and font credits:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Credits:** NVIDIA for Nemotron OCR v2 and the `nemotron-ocr` package; Angus
Johnson for Clipper; the ONNX, ONNX Runtime and PyTorch projects.
