# Running on each target

Every target uses the same steps:

1. Install the package (it builds the C++ glue ops CPU-only).
2. Get the ONNX graphs.
3. Install the right `onnxruntime` build.
4. Run `scripts/validate.py`.

Only step 3 differs between machines. Only the CPU column of the table below has
been run in this repository. The other rows are the documented setups for each
provider, so treat `validate.py` as the test on that machine.

| Target | ORT package | Provider | Tested here |
|---|---|---|---|
| CPU (x86-64, aarch64, Apple Silicon) | `onnxruntime` | `CPUExecutionProvider` | x86-64 Linux: yes |
| NVIDIA Tesla P40 (Pascal, sm_61) | `onnxruntime-gpu` < 1.27 (CUDA 12.x) | `CUDAExecutionProvider` | no |
| AMD Instinct / Radeon | AMD's `onnxruntime` MIGraphX build | `MIGraphXExecutionProvider` | no |
| macOS | `onnxruntime` | `CoreMLExecutionProvider` | no |
| Windows (any DX12 GPU) | `onnxruntime-directml` | `DmlExecutionProvider` | no |

All graphs are FP32 and use only standard ONNX ops (opset 17). If a provider can't
take a node, ORT runs that node on the CPU provider. The glue (image decode, NMS,
text/relation decoding) always runs on the CPU with PyTorch CPU. The PyTorch build
never needs CUDA or ROCm, so a CPU wheel is right on every target.

## Common steps

Python 3.11–3.13; the commands below use 3.12. On Windows, use
`py -3.12 -m venv .venv` and `.venv\Scripts\activate` instead.

```bash
git clone https://github.com/amffz9/portable-ocr-onnx.git
```

```bash
cd portable-ocr-onnx
```

```bash
python3.12 -m venv .venv && . .venv/bin/activate
```

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install numpy pillow "shapely>=2.1.2,<3" huggingface_hub setuptools ninja hatchling editables onnx pytest
```

```bash
pip install --no-build-isolation -e .
```

On macOS the PyTorch index above has no CPU-only wheels: drop
`--index-url ...`, because the default PyPI wheels are already CPU/MPS-only.
`--no-build-isolation` builds the extension against the torch you just installed.
It needs a C++ compiler:

- Linux: `g++`.
- macOS: Xcode command-line tools, plus `brew install libomp` for OpenMP. Without
  it the build still works, single-threaded.
- Windows: Visual Studio Build Tools with "Desktop development with C++".

Then get the ONNX graphs. There are two ways:

- **Export on this machine.** This needs the weights, about 3 GB of RAM and a few
  minutes of CPU. It runs the same on every OS and architecture.

  ```bash
  scripts/fetch_weights.sh all
  ```

  ```bash
  python scripts/export_onnx.py --variant english --no-check
  ```

  ```bash
  python scripts/export_onnx.py --variant multilingual --no-check
  ```

  On Windows without bash, run `hf download nvidia/nemotron-ocr-v2 --local-dir weights`
  instead of `fetch_weights.sh`.
- **Copy** the `onnx/` directory from a machine that already exported it. The
  graphs are platform-independent. You still need `weights/ocr-example-input-1.png`
  for validation, which `scripts/fetch_weights.sh` also downloads.

`--no-check` skips the tensor comparison, which needs the 2 GB full goldens
(`scripts/make_golden.py --full`). `validate.py` only needs the committed
`tests/golden/`.

Validate:

```bash
python scripts/validate.py --report validation_report.json
```

It prints the platform, Python, ORT version and available providers, plus the
provider each graph actually got. It reports the text match rate, the max box
error (must be ≤ 0.5 px), pages/s and peak memory. It exits 1 on any mismatch.

Use the model from Python:

```python
from nemotron_ocr_onnx import NemotronOCRONNX

ocr = NemotronOCRONNX("onnx/english", providers="auto")   # or ["CUDAExecutionProvider"], "cpu", ...
for p in ocr("page.png", merge_level="paragraph"):
    print(p["text"], p["left"], p["upper"], p["right"], p["lower"])
```

`ocr.session_providers` shows which provider each graph got (it is also logged at
INFO level). `providers="auto"` tries CUDA, MIGraphX, CoreML, DirectML and then CPU.
A provider named explicitly that the installed ORT lacks raises an error instead of
falling back silently.

## CPU (x86-64, aarch64 / ARM64, Apple Silicon)

```bash
pip install onnxruntime
```

```bash
python scripts/validate.py --providers cpu
```

- **aarch64 Linux** (Graviton, Ampere, Raspberry Pi 5, Jetson CPU): `onnxruntime`
  and the PyTorch CPU index both publish aarch64 wheels. The build uses no x86 flags
  there (`-mavx2` is only added on x86-64).
- **Windows on ARM**: `onnxruntime` has ARM64 wheels. PyTorch's Windows ARM64 wheels
  are newer and may not match the pinned 2.11 yet; this is the one target where the
  glue's PyTorch dependency may block (see "torch-free glue" in `PROGRESS.md`).
- `--threads N` sets ORT's intra-op threads. The default is all physical cores.

Reference numbers (4 vCPU x86-64, 1024 px): English 1.7 s/page, multilingual 1.9 s/page,
~3 GB peak RSS.

### Older and dual-socket Xeons (e.g. E5 v1–v4)

- **AVX2.** The C++ helpers are built with `-mavx2` only if the CPU you build on
  has AVX2. E5 v3/v4 (Haswell/Broadwell) have it; E5 v1/v2 (Sandy/Ivy Bridge) don't.
  Build on the machine that will run it, or set `NEMOTRON_OCR_AVX2=0` when building
  elsewhere for an AVX-only CPU. If `validate.py` dies with "Illegal instruction",
  report which package it came from.
- **Two sockets.** Threads spread across both sockets fight over memory traffic
  between them, which costs speed. Pin each OCR process to one socket and give it
  that socket's physical cores, e.g. for 8-core CPUs:

  ```bash
  numactl --cpunodebind=0 --membind=0 python scripts/validate.py --threads 8
  ```

  For throughput, run one worker process per socket (the second on node 1)
  instead of one process using every core. Install `numactl` from your
  distribution's packages; `lscpu` shows the cores per socket.
- **No VNNI or AVX-512.** These CPUs predate VNNI, so INT8 quantization gains
  little or nothing on them. If the server has a GPU with spare memory, the CUDA
  provider in FP32 is the faster route.

## NVIDIA Tesla P40 (Pascal) — CUDA provider, FP32

The P40 (compute capability 6.1) has no fast FP16, so FP32 is the right choice
anyway. Compatibility limits:

- CUDA 13 dropped Pascal. Since ORT 1.27 the PyPI `onnxruntime-gpu` wheels are
  built against CUDA 13, so use **1.26 or older** (CUDA 12.x, cuDNN 9).
- NVIDIA's cuDNN support matrix lists 9.10.2 as the last cuDNN for pre-Volta GPUs.
- Driver: any R525+ driver that supports CUDA 12.x (`nvidia-smi` shows the highest
  CUDA version it supports).
- PyTorch itself isn't used on the GPU here. (PyTorch's own Pascal support ends
  with 2.14 / CUDA 12.6, which only matters for the PyTorch reference pipeline.)

```bash
pip uninstall -y onnxruntime
```

```bash
pip install "onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"
```

```bash
pip install "nvidia-cudnn-cu12==9.10.2.*"
```

```bash
python -c "import onnxruntime as o; o.preload_dlls(); print(o.get_available_providers())"
```

```bash
python scripts/validate.py --providers cuda --report p40.json
```

- The `[cuda,cudnn]` extras pull the CUDA 12 runtime, cuBLAS and cuDNN as pip
  packages. A system CUDA toolkit install isn't needed.
- If `CUDAExecutionProvider` is listed but sessions fall back to CPU, run with `-v`
  to see ORT's error. `nemotron_ocr_onnx` calls `onnxruntime.preload_dlls()` when
  CUDA is requested, which loads the pip-installed CUDA/cuDNN libraries. If
  `libcudnn*.so.9` / `libcublas*.so.12` are still not found, set `LD_LIBRARY_PATH`
  to the `nvidia/*/lib` directories in site-packages.
- If a run fails with "no kernel image is available for execution on the device",
  that ORT wheel wasn't compiled for sm_61. Try an older 1.2x release, or build ORT
  from source with `--cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=61`.
- The P40 has 24 GB. The defaults (detector batch 8, recognizer chunk 32) fit easily.
  Raise `detector_max_batch_size` for throughput on many pages.
- Other GPUs: on Volta or newer you can use the current `onnxruntime-gpu`; the
  steps are the same without the version pins.

## AMD GPUs — MIGraphX provider

ONNX Runtime removed its ROCm execution provider in 1.23; MIGraphX is AMD's
supported path. AMD ships the builds, and the package name and index change
between ROCm releases, so follow the page for your ROCm version: [ONNX Runtime on
ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html)
(Instinct) or "Install ONNX Runtime for Radeon GPUs" in the ROCm Radeon docs.
In outline:

1. Install ROCm and MIGraphX for your GPU, per AMD's docs.
2. Remove other ORT builds: `pip uninstall -y onnxruntime onnxruntime-gpu`.
3. Install AMD's MIGraphX-enabled ORT wheel from the URL or index on that page.
   Older Radeon wheels need `numpy<2`; recent ROCm docs use a separate
   `onnxruntime-ep-migraphx` plugin that must be registered before use.
4. Check that `python -c "import onnxruntime as o; print(o.get_available_providers())"`
   lists `MIGraphXExecutionProvider`.
5. Run the validation:

   ```bash
   python scripts/validate.py --providers migraphx --report amd.json
   ```

Notes:

- MIGraphX compiles a program for each new input shape. The detector input is
  always `[B, 3, 1024, 1024]`, but region counts vary per page, so the sampler,
  recognizer and relational graphs recompile as counts change. The first pages are
  slow, and `validate.py`'s warm-up page only covers the first shapes.

## macOS — CoreML provider

```bash
pip install onnxruntime
```

```bash
python scripts/validate.py --providers coreml --report mac.json
```

- The standard macOS `onnxruntime` wheel includes `CoreMLExecutionProvider`.
- CoreML takes the parts of each graph it supports and leaves the rest on CPU. With
  dynamic region counts, much of the sampler, recognizer and relational graphs may
  stay on CPU. Compare `--providers coreml` against `--providers cpu` and keep the
  faster one. On Apple Silicon the CPU provider is already fast.
- Tolerance note: CoreML may compute in reduced precision on the Neural Engine. If
  validation fails only on CoreML, report it; the CPU provider is the fallback.
- OpenMP for the glue: `brew install libomp` before `pip install -e .` (optional).

## Windows — DirectML provider

```powershell
pip uninstall -y onnxruntime
pip install onnxruntime-directml
python scripts\validate.py --providers dml --report dml.json
```

- Works with any DirectX 12 GPU (NVIDIA, AMD, Intel) on Windows 10 1903+.
- `nemotron_ocr_onnx` sets the session options DirectML requires (memory pattern
  off, sequential execution).
- The C++ glue builds with MSVC (`/openmp`). Run the `pip install -e .` step from a
  "Developer PowerShell for VS" if `cl.exe` isn't found.

## CPU fallback everywhere

If no accelerator works, `--providers cpu` (or `providers="cpu"` in Python) always
works with the plain `onnxruntime` wheel. A GPU provider that fails to create a
session falls back to CPU with a warning. `validate.py` prints the provider each
graph actually ran on, so a silent fallback is visible.
