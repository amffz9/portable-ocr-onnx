"""NemotronOCRONNX end-to-end on the upstream sample page, and provider resolution.

The full comparison over every golden page is scripts/validate.py.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "onnx" / "english"
SAMPLE = ROOT / "weights" / "ocr-example-input-1.png"

ort = pytest.importorskip("onnxruntime")


def test_resolve_providers():
    from nemotron_ocr_onnx import resolve_providers

    auto = resolve_providers("auto")
    assert auto[-1] == "CPUExecutionProvider"
    assert set(auto) <= set(ort.get_available_providers())
    assert resolve_providers("cpu") == ["CPUExecutionProvider"]
    with pytest.raises(ValueError):
        resolve_providers(["NoSuchExecutionProvider"])


@pytest.mark.skipif(not (ONNX_DIR / "detector.onnx").exists() or not SAMPLE.exists(),
                    reason="needs onnx/english/ and weights/ocr-example-input-1.png")
def test_sample_matches_golden():
    from nemotron_ocr_onnx import NemotronOCRONNX

    ocr = NemotronOCRONNX(ONNX_DIR, providers="cpu")
    assert set(ocr.session_providers.values()) == {"CPUExecutionProvider"}
    ref = json.loads((ROOT / "tests" / "golden" / "english" / "sample.json").read_text())
    h, w = ref["original_hw"]
    preds = ocr(str(SAMPLE), merge_level="paragraph")
    golden = ref["predictions"]["paragraph"]
    assert [p["text"] for p in preds] == [g["text"] for g in golden]
    for p, g in zip(preds, golden):
        box = [p["left"] * w, p["lower"] * h, p["right"] * w, p["upper"] * h]
        assert max(abs(a - b) for a, b in zip(box, g["box_px"])) <= 0.5
