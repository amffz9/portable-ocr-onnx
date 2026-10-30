"""ONNX Runtime execution provider selection."""

import logging

import onnxruntime as ort

logger = logging.getLogger(__name__)

# Preference order for providers="auto". Unavailable providers are skipped; CPU is
# always last so every graph (or partition a GPU provider can't take) still runs.
AUTO_ORDER = (
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)

ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
}


def resolve_providers(providers="auto"):
    """Return the ORT provider list to request.

    providers: "auto", a single name or alias ("cuda", "CPUExecutionProvider"), or a
    list of names / (name, options) tuples as accepted by ``ort.InferenceSession``.
    Explicitly requested providers that this onnxruntime build lacks raise an error
    rather than silently falling back.
    """
    available = ort.get_available_providers()
    if providers is None or providers == "auto":
        chosen = [p for p in AUTO_ORDER if p in available]
        # Providers outside AUTO_ORDER (e.g. TensorRT, OpenVINO) need explicit opt-in.
        return chosen or ["CPUExecutionProvider"]

    if isinstance(providers, str):
        providers = [providers]
    out = []
    for p in providers:
        name, opts = (p if isinstance(p, tuple) else (p, None))
        name = ALIASES.get(name.lower(), name) if isinstance(name, str) else name
        if name not in available:
            raise ValueError(f"{name} is not available in this onnxruntime build "
                             f"(available: {', '.join(available)})")
        out.append((name, opts) if opts else name)
    if "CPUExecutionProvider" not in [p[0] if isinstance(p, tuple) else p for p in out]:
        out.append("CPUExecutionProvider")
    return out


def _names(providers):
    return [p[0] if isinstance(p, tuple) else p for p in providers]


def session_options(providers, intra_op_threads=0):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads:
        so.intra_op_num_threads = intra_op_threads
    if "DmlExecutionProvider" in _names(providers):
        # Required by the DirectML provider.
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def preload_cuda_libraries(providers):
    """Load the pip-installed CUDA / cuDNN libraries (onnxruntime-gpu >= 1.21)."""
    if "CUDAExecutionProvider" in _names(providers) and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:  # noqa: BLE001 - system CUDA may still be on the path
            logger.info("onnxruntime.preload_dlls() failed: %s", e)


def create_session(path, providers, intra_op_threads=0):
    """Create a session; if a GPU provider fails to initialise, retry on CPU."""
    so = session_options(providers, intra_op_threads)
    try:
        sess = ort.InferenceSession(str(path), so, providers=providers)
    except Exception as e:  # noqa: BLE001 - provider init errors vary by build
        if _names(providers) == ["CPUExecutionProvider"]:
            raise
        logger.warning("%s: %s failed (%s); falling back to CPUExecutionProvider",
                       path.name, _names(providers)[0], e)
        sess = ort.InferenceSession(str(path), session_options(["CPUExecutionProvider"], intra_op_threads),
                                    providers=["CPUExecutionProvider"])
    return sess
