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

Work in progress; see [PROGRESS.md](PROGRESS.md) for what is done.

## Quick start (PyTorch reference pipeline, CPU)

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
- `scripts/fetch_weights.sh`: weight download.

## Licences

Code: Apache 2.0 (`LICENSE`, from upstream). Model weights: NVIDIA Open Model
License (`NOTICE`); they're downloaded from Hugging Face and never committed. This
project isn't affiliated with or endorsed by NVIDIA.
