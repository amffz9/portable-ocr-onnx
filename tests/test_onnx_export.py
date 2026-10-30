"""Edge cases for the exported graphs that the golden pages don't cover.

Compares ONNX Runtime (CPU) against the upstream PyTorch / C++ CPU code. Skipped if
the graphs or weights are missing (run scripts/fetch_weights.sh and
scripts/export_onnx.py --variant english first).
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "onnx" / "english"
WEIGHTS = ROOT / "weights" / "v2_english"

ort = pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.skipif(
    not (ONNX_DIR / "relational.onnx").exists() or not (WEIGHTS / "detector.pth").exists(),
    reason="needs weights/ and onnx/english/",
)


@pytest.fixture(scope="module")
def base():
    from nemotron_ocr.inference.pipeline import NemotronOCR

    warnings.filterwarnings("ignore")
    m = NemotronOCR(model_dir=str(WEIGHTS), device="cpu", precision="fp32")
    for net in (m.detector, m.recognizer, m.relational):
        net.eval()
    return m


def session(name):
    return ort.InferenceSession(str(ONNX_DIR / f"{name}.onnx"), providers=["CPUExecutionProvider"])


def random_quads(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    tl = torch.rand(n, 2, generator=g) * 800 + 50
    wh = torch.rand(n, 2, generator=g) * torch.tensor([150.0, 30.0]) + torch.tensor([20.0, 10.0])
    return torch.stack((tl, tl + wh * torch.tensor([1.0, 0.0]), tl + wh, tl + wh * torch.tensor([0.0, 1.0])), 1)


@pytest.mark.parametrize("n", [1, 2, 15, 16, 17])
def test_relational_small_n(base, n):
    g = torch.Generator().manual_seed(n)
    rel_in = torch.randn(n, 128, 2, 3, generator=g)
    quads = random_quads(n, seed=n)
    rec_feat = torch.randn(n, base.max_width, base.recognizer.feature_depth, generator=g)
    with torch.no_grad():
        ref = base.relational(rel_in, quads, torch.tensor([n]), rec_feat)
    words, lines, var = session("relational").run(
        None, {"rel_input": rel_in.numpy(), "quads": quads.numpy(), "rec_features": rec_feat.numpy()}
    )
    for got, key in ((words, "words"), (lines, "lines"), (var, "line_log_var_unc")):
        r = ref[key][0].numpy()
        assert got.shape == r.shape == (n, n + 1)
        assert np.array_equal(np.isinf(got), np.isinf(r))
        fin = np.isfinite(r)
        np.testing.assert_allclose(got[fin], r[fin], atol=5e-4, rtol=1e-4)


def test_sampler_second_image_of_batch(base):
    from nemotron_ocr_cpp import indirect_grid_sample_forward, quad_rectify_forward

    g = torch.Generator().manual_seed(1)
    feats = torch.randn(2, 128, 192, 160, generator=g)  # non-square, batch of 2
    quads = random_quads(9, seed=3) * 0.6
    h, w = 192 * 4, 160 * 4
    rec_in, rel_in = session("sampler").run(
        None, {"features": feats.numpy(), "image_index": np.array([1], dtype=np.int64), "quads": quads.numpy()}
    )
    idx = torch.ones(9, dtype=torch.int64)
    ref_rec = indirect_grid_sample_forward(feats, quad_rectify_forward(quads, h, w, 8, base.max_width, True), idx, "bilinear")
    ref_rel = indirect_grid_sample_forward(feats, quad_rectify_forward(quads, h, w, 2, 3, False), idx, "bilinear")
    np.testing.assert_allclose(rec_in, ref_rec.numpy(), atol=1e-5)
    np.testing.assert_allclose(rel_in, ref_rel.numpy(), atol=1e-5)


def test_detector_dynamic_shape(base):
    x = torch.rand(2, 3, 512, 768, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        conf, _, rboxes, feats = base.detector(x)
    got = session("detector").run(None, {"image": x.numpy()})
    for a, b in zip(got, (conf, rboxes, feats)):
        assert a.shape == tuple(b.shape)
        np.testing.assert_allclose(a, b.numpy(), atol=2e-3, rtol=1e-4)
