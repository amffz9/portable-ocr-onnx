# portable-ocr-onnx

A portable build of NVIDIA's Nemotron OCR v2 (`nemotron-ocr` 2.0.2) that runs
without CUDA:

- The three networks (detector, recognizer, relational model) are exported to ONNX
  and run through ONNX Runtime, so the same model files work on CPU, NVIDIA CUDA
  (including Pascal cards such as the Tesla P40), AMD (MIGraphX), Apple (CoreML) and
  Windows (DirectML).
- The C++ glue ops (NMS, quad rectification, decoding) build CPU-only, with no
  `nvcc` or CUDA toolkit.
- A validation harness checks that the ONNX pipeline matches the original PyTorch
  pipeline on any target machine.

Status: all milestones done and validated on CPU (x86-64 Linux). GPU and ARM
targets have documented steps but haven't been run yet. See
[PROGRESS.md](PROGRESS.md) and [docs/TARGETS.md](docs/TARGETS.md).

## Quick start (ONNX Runtime)

Install as in the next section, then:

```bash
pip install onnxruntime onnx              # or onnxruntime-gpu / -directml, see docs/TARGETS.md
python scripts/export_onnx.py --variant english --no-check   # writes onnx/english/
python scripts/validate.py --variants english                 # compares against tests/golden
```

```python
from nemotron_ocr_onnx import NemotronOCRONNX

ocr = NemotronOCRONNX("onnx/english", providers="auto")
print(ocr("page.png", merge_level="paragraph"))
```

## Install and run the PyTorch reference pipeline (CPU)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow "shapely>=2.1.2,<3" huggingface_hub setuptools ninja hatchling editables
pip install --no-build-isolation -e .   # builds the CPU-only extension when CUDA isn't available
scripts/fetch_weights.sh                # downloads the checkpoints into weights/
python example.py weights/ocr-example-input-1.png --model-dir weights/v2_english
```

Build options (environment variables): `NEMOTRON_OCR_CPU_ONLY=1` (skip CUDA even if
`nvcc` is present), `NEMOTRON_OCR_NO_OPENMP=1`, `NEMOTRON_OCR_LIBOMP_PREFIX` (macOS),
`BUILD_CPP_FORCE=1` (rebuild).

## Layout

- `src/nemotron_ocr/`, `src/nemotron_ocr_cpp/`, `cpp/`: upstream package, modified
  for CPU builds (each changed file says so at the top).
- `docs/ARCHITECTURE.md`: how the upstream pipeline fits together.
- `src/nemotron_ocr_onnx/`: ONNX export wrappers (`export_modules.py`) and the
  ONNX Runtime pipeline (`NemotronOCRONNX`).
- `scripts/fetch_weights.sh`: weight download.
- `scripts/export_onnx.py`: export + check against the full goldens.
- `scripts/make_test_pages.py`, `scripts/make_golden.py`: test pages and reference
  outputs (`tests/images/`, `tests/golden/`).
- `scripts/validate.py`: end-to-end validation on any machine.
- `docs/TARGETS.md`: install steps per platform.

## Licences

Code: Apache 2.0 (`LICENSE`, from upstream). Model weights: NVIDIA Open Model
License (`NOTICE`); they're downloaded from Hugging Face and never committed. This
project isn't affiliated with or endorsed by NVIDIA.
