"""Tests for the pure-PyTorch port of ragged_quad_all_2_all_distance_v2."""

import math

import pytest
import torch

from nemotron_ocr.inference.models.quad_distance import (
    quad_all_2_all_distance,
    quad_all_2_all_distance_reference,
    ragged_quad_all_2_all_distance_v2_torch,
)


def box(x0, y0, x1, y1):
    """Axis-aligned quad in the detector's vertex order: TL, TR, BR, BL."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


IMPLS = [quad_all_2_all_distance, quad_all_2_all_distance_reference]


@pytest.mark.parametrize("fn", IMPLS)
def test_hand_computed_axis_aligned(fn):
    quads = torch.tensor(
        [
            box(0, 0, 10, 2),      # A
            box(13, 0, 20, 2),     # B: 3 to the right of A
            box(0, 6, 10, 8),      # C: 4 below A
            box(14, 6, 20, 8),     # D: diagonal from A: dx=4, dy=4
        ],
        dtype=torch.float32,
    )
    d = fn(quads, allow_self_distance=False)
    assert torch.isinf(d.diagonal()).all()
    expect = {
        (0, 1): 3.0,
        (0, 2): 4.0,
        (0, 3): math.hypot(4, 4),
        (1, 2): math.hypot(3, 4),
        (1, 3): 4.0,
        (2, 3): 4.0,
    }
    for (i, j), v in expect.items():
        assert d[i, j].item() == pytest.approx(v, abs=1e-5), (i, j)
        assert d[j, i].item() == pytest.approx(v, abs=1e-5), (j, i)


@pytest.mark.parametrize("fn", IMPLS)
def test_self_distance_and_rotated(fn):
    # A quad rotated 90 degrees: the frame changes but distances are rotation-invariant.
    a = box(0, 0, 4, 1)
    b = [[10, 0], [10, 4], [9, 4], [9, 0]]  # vertical box spanning x in [9, 10]
    d = fn(torch.tensor([a, b], dtype=torch.float32), allow_self_distance=True)
    assert d[0, 0].item() == 0 and d[1, 1].item() == 0
    assert d[0, 1].item() == pytest.approx(5.0, abs=1e-5)
    assert d[1, 0].item() == pytest.approx(5.0, abs=1e-5)


@pytest.mark.parametrize("fn", IMPLS)
def test_crossing_edges_use_endpoint_distance(fn):
    # Two boxes forming a plus sign: edges cross, but the kernel only measures
    # endpoint-to-segment distances, so the result is NOT zero. The closest
    # vertex/edge pair is the vertical bar's vertex (4, -2) to the horizontal bar's
    # top edge y=0: distance 2.
    h = box(0, 0, 10, 4)
    v = box(4, -2, 6, 8)
    d = fn(torch.tensor([h, v], dtype=torch.float32), allow_self_distance=False)
    assert d[0, 1].item() == pytest.approx(2.0, abs=1e-5)
    assert d[1, 0].item() == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize("fn", IMPLS)
def test_degenerate_quad(fn):
    # Zero-length frame vector and zero-length edges must not produce NaN.
    pt = [[5, 5]] * 4
    d = fn(torch.tensor([pt, box(8, 5, 9, 6)], dtype=torch.float32), allow_self_distance=False)
    assert torch.isfinite(d[0, 1]) and d[0, 1].item() == pytest.approx(3.0, abs=1e-5)


def test_fast_matches_reference_random():
    g = torch.Generator().manual_seed(0)
    for n in (1, 2, 7, 40):
        centers = torch.rand(n, 1, 2, generator=g) * 1000
        sizes = torch.rand(n, 1, 2, generator=g) * 80 + 2
        ang = torch.rand(n, generator=g) * 2 * math.pi
        base = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        pts = base[None] * sizes
        rot = torch.stack(
            [torch.stack([ang.cos(), -ang.sin()], -1), torch.stack([ang.sin(), ang.cos()], -1)], -2
        )
        quads = torch.einsum("nij,nkj->nki", rot, pts) + centers
        quads = quads + torch.randn(n, 4, 2, generator=g) * 2  # irregular quads
        ref = quad_all_2_all_distance_reference(quads, allow_self_distance=False)
        fast = quad_all_2_all_distance(quads, allow_self_distance=False)
        assert torch.equal(torch.isinf(ref), torch.isinf(fast))
        fin = torch.isfinite(ref)
        torch.testing.assert_close(fast[fin], ref[fin], rtol=1e-5, atol=1e-4)


def test_x_y_factors_reference_semantics():
    # Positive components along the row quad's frame are scaled.
    a = box(0, 0, 2, 2)
    b = box(5, 0, 7, 2)   # 3 to the +x side of a
    c = box(-5, 0, -3, 2)  # 3 to the -x side of a
    d = quad_all_2_all_distance(
        torch.tensor([a, b, c], dtype=torch.float32), x_factor=2.0, allow_self_distance=False
    )
    assert d[0, 1].item() == pytest.approx(6.0, abs=1e-5)
    assert d[0, 2].item() == pytest.approx(3.0, abs=1e-5)


def test_ragged_batch_layout():
    q = torch.zeros(2, 3, 4, 2)
    q[0, 0] = torch.tensor(box(0, 0, 1, 1), dtype=torch.float32)
    q[0, 1] = torch.tensor(box(3, 0, 4, 1), dtype=torch.float32)
    q[1, 0] = torch.tensor(box(0, 0, 1, 1), dtype=torch.float32)
    out = ragged_quad_all_2_all_distance_v2_torch(q, torch.tensor([2, 1]), allow_self_distance=False)
    assert out.shape == (2, 3, 3)
    assert out[0, 0, 1].item() == pytest.approx(2.0)
    assert torch.isinf(out[0, 0, 0]) and torch.isinf(out[1, 0, 0])
    assert out[0, 2].abs().sum() == 0 and out[1, 1:].abs().sum() == 0
