"""NemotronOCRONNX: the NemotronOCRV2 pipeline with its networks run by ONNX Runtime.

The four ONNX graphs (detector, sampler, recognizer, relational) replace the PyTorch
networks and the C++ quad-rectify / grid-sample ops. Image loading, the NMS
prefilter, NMS (C++ ``nemotron_ocr_cpp``), and text / relation decoding stay as in
upstream ``NemotronOCRV2`` and run on CPU with torch. Only CPU tensors cross between
the two, so any ONNX Runtime execution provider works.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from nemotron_ocr.inference.pipeline_v2 import (
    _DEFAULT_PAD_COLOR,
    DEFAULT_MERGE_LEVEL,
    NemotronOCRV2,
)

from .providers import create_session, preload_cuda_libraries, resolve_providers

logger = logging.getLogger(__name__)

GRAPHS = ("detector", "sampler", "recognizer", "relational")


class NemotronOCRONNX(NemotronOCRV2):
    """OCR pipeline backed by ONNX Runtime; same call API as ``NemotronOCRV2``.

    Args:
        model_dir: directory written by ``scripts/export_onnx.py``
            (``onnx/english`` or ``onnx/multilingual``).
        providers: ``"auto"`` (CUDA, MIGraphX, CoreML, DirectML, then CPU; whichever
            this onnxruntime build has) or an explicit provider list, e.g.
            ``["CUDAExecutionProvider"]``, ``"cpu"``, or ``[("CUDAExecutionProvider",
            {"device_id": 1})]``. CPU is always appended as the fallback.
        intra_op_threads: ORT CPU thread count (0 = ORT default).
        detector_max_batch_size: images per detector run. Default 1 when the
            detector runs on the CPU provider (batching gives no speed-up there and
            costs ~800 MB per extra image) and 8 otherwise, as upstream.
        recognizer_chunk_size: regions per recognizer run. The recognizer also
            materialises logits ([chunk, 128, vocab] fp32, ~7 MB per region for the
            multilingual vocabulary) inside ORT, so this bounds peak memory.
        use_prefilter, prefilter_peak_kernel, skip_relational, detector_only,
        pad_color, pad_how, infer_length, verbose_post: as in ``NemotronOCRV2``.
    """

    def __init__(
        self,
        model_dir,
        *,
        providers="auto",
        intra_op_threads: int = 0,
        detector_max_batch_size=None,
        recognizer_chunk_size: int = 32,
        use_prefilter: bool = True,
        prefilter_peak_kernel: int = 3,
        detector_only: bool = False,
        skip_relational: bool = False,
        pad_color=None,
        pad_how=None,
        infer_length=None,
        verbose_post: bool = False,
    ):
        # Glue ops run on CPU in FP32; the graphs are FP32.
        self._setup_device("cpu", "fp32")
        self._detector_only = detector_only
        self._skip_relational = skip_relational
        self._model_dir = Path(model_dir)
        missing = [g for g in GRAPHS if not (self._model_dir / f"{g}.onnx").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{self._model_dir} is missing {', '.join(m + '.onnx' for m in missing)}; "
                "run scripts/export_onnx.py first")

        self.meta = json.loads((self._model_dir / "model_meta.json").read_text())
        self._load_config(infer_length_override=infer_length)
        self._load_charset()
        self.max_width = self.meta["max_width"]
        self.num_tokens = self.meta["num_tokens"]
        self._load_sessions(providers, intra_op_threads)
        self._initialize_processors()

        if detector_max_batch_size is None:
            on_cpu = self.session_providers["detector"] == "CPUExecutionProvider"
            detector_max_batch_size = 1 if on_cpu else 8
        self.detector_max_batch_size = detector_max_batch_size
        self.recognizer_chunk_size = recognizer_chunk_size
        self._use_prefilter = use_prefilter
        self._prefilter_peak_kernel = prefilter_peak_kernel
        self._verbose_post = verbose_post
        self._pad_color_cpu = (torch.tensor(pad_color, dtype=torch.float16) if pad_color is not None
                               else _DEFAULT_PAD_COLOR.clone())
        self._pad_color = None
        self._pad_how = pad_how or "bottom_right"

    # ------------------------------------------------------------------ sessions

    def _load_sessions(self, providers, intra_op_threads):
        requested = resolve_providers(providers)
        preload_cuda_libraries(requested)
        self.sessions = {}
        self.session_providers = {}
        needed = ["detector"]
        if not self._detector_only:
            needed += ["sampler", "recognizer"]
            if not self._skip_relational:
                needed.append("relational")
        for g in needed:
            sess = create_session(self._model_dir / f"{g}.onnx", requested, intra_op_threads)
            self.sessions[g] = sess
            self.session_providers[g] = sess.get_providers()[0]
            logger.info("%s: %s", g, self.session_providers[g])

    def _load_models(self):  # PyTorch weights are not used
        raise NotImplementedError

    @staticmethod
    def _np(x):
        return np.ascontiguousarray(x.detach().cpu().numpy())

    # ------------------------------------------------------------------ phases

    def _run_detector_batched(self, resized_batch):
        sess = self.sessions["detector"]
        x = self._np(resized_batch.float())
        parts = [sess.run(None, {"image": x[s:s + self.detector_max_batch_size]})
                 for s in range(0, x.shape[0], self.detector_max_batch_size)]
        conf, rboxes, feats = (np.concatenate(p, 0) if len(parts) > 1 else p[0] for p in zip(*parts))
        return torch.from_numpy(conf), torch.from_numpy(rboxes), torch.from_numpy(feats)

    def _run_rectify_and_sample(self, quads, region_counts, det_features):
        sess = self.sessions["sampler"]
        feats = self._np(det_features)
        quads_np = self._np(quads.float())
        counts = region_counts.cpu()
        rec_parts, rel_parts = [], []
        offset = 0
        for i, n in enumerate(counts.tolist()):
            if n == 0:
                continue
            rec, rel = sess.run(None, {
                "features": feats[i:i + 1],
                "image_index": np.zeros(1, dtype=np.int64),
                "quads": quads_np[offset:offset + n],
            })
            rec_parts.append(rec)
            rel_parts.append(rel)
            offset += n
        rec_quads = torch.from_numpy(np.concatenate(rec_parts, 0))
        rel_quads = None if self._skip_relational else torch.from_numpy(np.concatenate(rel_parts, 0))
        return quads, rec_quads, rel_quads, counts

    def _run_recognizer_chunked(self, rec_quads):
        n = rec_quads.shape[0]
        depth = self.meta["recognizer_feature_depth"]
        if n == 0:
            return (torch.empty(0, self.max_width, dtype=torch.int64),
                    torch.empty(0, self.max_width, dtype=torch.float32),
                    torch.empty(0, self.max_width, depth, dtype=torch.float32))
        sess = self.sessions["recognizer"]
        x = self._np(rec_quads.float())
        cs = self.recognizer_chunk_size
        outs = [sess.run(["token_ids", "token_probs", "features"], {"rec_input": x[s:s + cs]})
                for s in range(0, n, cs)]
        ids, probs, feats = (np.concatenate(p, 0) for p in zip(*outs))
        return torch.from_numpy(ids), torch.from_numpy(probs), torch.from_numpy(feats)

    def relational(self, rel_quads, quads, region_counts, rec_features):
        """Same interface and outputs as ``GlobalRelationalModel.forward``."""
        sess = self.sessions["relational"]
        rel_np, quads_np, feat_np = self._np(rel_quads), self._np(quads.float()), self._np(rec_features)
        out = {"words": [], "lines": [], "line_log_var_unc": []}
        offset = 0
        for n in region_counts.tolist():
            if n == 0:
                for v in out.values():
                    v.append(torch.empty(0, 1, dtype=torch.float32))
                continue
            sl = slice(offset, offset + n)
            words, lines, var = sess.run(None, {
                "rel_input": rel_np[sl], "quads": quads_np[sl], "rec_features": feat_np[sl]})
            out["words"].append(torch.from_numpy(words))
            out["lines"].append(torch.from_numpy(lines))
            out["line_log_var_unc"].append(torch.from_numpy(var))
            offset += n
        return out

    def __call__(self, images, merge_level: str = DEFAULT_MERGE_LEVEL, include_invalid: bool = False):
        with torch.inference_mode():
            return super().__call__(images, merge_level=merge_level, include_invalid=include_invalid)
