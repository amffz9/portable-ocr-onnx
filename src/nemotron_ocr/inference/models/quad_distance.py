"""Pure-PyTorch port of the CUDA-only ``ragged_quad_all_2_all_distance_v2`` op.

Part of the portable ONNX port of Nemotron OCR v2. The upstream op lives in
``cpp/local_ips/quad_all_2_all_dist_v2.cu`` and has no CPU implementation. This
module reproduces its math with standard tensor ops so it runs on any device and
exports to ONNX.

For a pair of quads (row, col) the kernel computes, over all 4x4 pairs of edges
(row edge i: v[i] -> v[(i+1)%4], col edge j likewise):

1. ``shortest_line_between_segments(rowEdge, colEdge)`` from ``cpp/geometry.h``.
   That considers only the four endpoint-to-segment candidates
   (rowEdge.A -> colEdge, rowEdge.B -> colEdge, rowEdge -> colEdge.A,
   rowEdge -> colEdge.B), each given as a vector, and keeps the first one with
   the strictly smallest squared length. Crossing edges are *not* detected, so two
   intersecting edges still report the endpoint distance, not 0.
2. The winning vector is re-projected into the row quad's frame: ``vX`` is the unit
   vector from the midpoint of edge (v0, v3) to the midpoint of edge (v1, v2)
   (``(1, 0)`` if that length is 0), ``vY = (-vX.y, vX.x)``. Positive components
   are scaled by ``x_factor`` / ``y_factor``.
3. The distance is the length of the re-projected vector; the result is the min over
   the 16 edge pairs.

The diagonal is 0 if ``allow_self_distance`` else +inf. Entries beyond each
example's region count are 0, as in the CUDA output tensor (``torch::zeros``).

Two implementations:

- :func:`quad_all_2_all_distance_reference` follows the kernel step by step,
  including the per-edge-pair candidate selection. It materialises
  [N, N, 4, 4, 4, 2] and is meant for testing.
- :func:`quad_all_2_all_distance` is what the pipeline and the ONNX export use. With
  ``x_factor == y_factor == 1`` (the only values upstream passes) step 2 is a pure
  rotation, which preserves length, so the result is the minimum vertex-to-edge
  distance in either direction: ``min(D[r, c], D[c, r])`` where ``D[r, c]`` is the
  smallest distance from a vertex of quad r to an edge of quad c. That is the same
  value in exact arithmetic, differs only by float rounding (~1 ulp), and needs
  [N, N, 4, 4] memory. Other factors fall back to the reference implementation.
"""

import torch


def _nearest_point_on_segment(pt: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``nearest_point_on_segment`` from geometry.h, broadcast over leading dims.

    pt, a, b: [..., 2]. Returns the closest point on segment (a, b) to pt.
    """
    w_minus_v = b - a
    l2 = (w_minus_v * w_minus_v).sum(-1, keepdim=True)
    t = ((pt - a) * w_minus_v).sum(-1, keepdim=True) / torch.where(l2 == 0, torch.ones_like(l2), l2)
    t = t.clamp(0.0, 1.0)
    proj = a + t * w_minus_v
    return torch.where(l2 == 0, a.expand_as(proj), proj)


def quad_all_2_all_distance_reference(
    quads: torch.Tensor,
    x_factor: float = 1.0,
    y_factor: float = 1.0,
    allow_self_distance: bool = True,
) -> torch.Tensor:
    """All-to-all quad distances for a single example.

    Args:
        quads: [N, 4, 2] float tensor.
    Returns:
        [N, N] distances (row quad defines the re-projection frame).
    """
    quads = quads.float()
    n = quads.shape[0]

    # Row-quad frame
    p1 = (quads[:, 0] + quads[:, 3]) / 2.0
    p2 = (quads[:, 1] + quads[:, 2]) / 2.0
    vx = p2 - p1
    len_vx = vx.norm(dim=-1, keepdim=True)
    vx = torch.where(
        len_vx > 0,
        vx / len_vx.clamp_min(1e-8),
        torch.tensor([1.0, 0.0], dtype=quads.dtype, device=quads.device).expand_as(vx),
    )
    vy = torch.stack((-vx[:, 1], vx[:, 0]), dim=-1)

    # Edges: [N, 4, 2] start/end points
    e_a = quads
    e_b = torch.roll(quads, shifts=-1, dims=1)

    # Broadcast to [N(row), N(col), 4(row edge), 4(col edge), 2]
    ra = e_a[:, None, :, None, :]
    rb = e_b[:, None, :, None, :]
    ca = e_a[None, :, None, :, :]
    cb = e_b[None, :, None, :, :]
    shape = (n, n, 4, 4, 2)
    ra, rb, ca, cb = (t.expand(shape) for t in (ra, rb, ca, cb))

    # The four candidate segments, as (B - A) vectors, in geometry.h order
    cands = torch.stack(
        (
            _nearest_point_on_segment(ra, ca, cb) - ra,
            _nearest_point_on_segment(rb, ca, cb) - rb,
            ca - _nearest_point_on_segment(ca, ra, rb),
            cb - _nearest_point_on_segment(cb, ra, rb),
        ),
        dim=-2,
    )  # [N, N, 4, 4, 4(cand), 2]

    # First strict minimum of squared length (argmin returns the first on ties)
    lsq = (cands * cands).sum(-1)
    idx = lsq.argmin(dim=-1, keepdim=True)
    v_seg = torch.gather(cands, -2, idx.unsqueeze(-1).expand(*idx.shape, 2)).squeeze(-2)

    # Re-project into the row quad's frame
    vx_b = vx[:, None, None, None, :]
    vy_b = vy[:, None, None, None, :]
    d_x = (v_seg * vx_b).sum(-1)
    d_y = (v_seg * vy_b).sum(-1)
    if x_factor != 1.0:
        d_x = torch.where(d_x >= 0, d_x * x_factor, d_x)
    if y_factor != 1.0:
        d_y = torch.where(d_y >= 0, d_y * y_factor, d_y)
    dist = torch.sqrt(d_x * d_x + d_y * d_y)

    dist = dist.flatten(2).amin(dim=-1)  # [N, N]

    diag_val = 0.0 if allow_self_distance else float("inf")
    eye = torch.eye(n, dtype=torch.bool, device=quads.device)
    return torch.where(eye, torch.full_like(dist, diag_val), dist)


def _vertex_to_edge_sq_dist(quads: torch.Tensor) -> torch.Tensor:
    """D2[r, c] = min squared distance from any vertex of quad r to any edge of quad c."""
    pts = quads[:, None, :, None, :]                     # [N, 1, 4(vertex), 1, 2]
    a = quads[None, :, None, :, :]                       # [1, N, 1, 4(edge), 2]
    b = torch.roll(quads, shifts=-1, dims=1)[None, :, None, :, :]
    ab = b - a
    l2 = (ab * ab).sum(-1)                               # [1, N, 1, 4]
    t = ((pts - a) * ab).sum(-1) / torch.where(l2 == 0, torch.ones_like(l2), l2)
    t = torch.where(l2 == 0, torch.zeros_like(t), t.clamp(0.0, 1.0))
    diff = a + t.unsqueeze(-1) * ab - pts                # [N, N, 4, 4, 2]
    return (diff * diff).sum(-1).flatten(2).amin(dim=-1)  # [N, N]


def quad_all_2_all_distance(
    quads: torch.Tensor,
    x_factor: float = 1.0,
    y_factor: float = 1.0,
    allow_self_distance: bool = True,
) -> torch.Tensor:
    """All-to-all quad distances for a single example ([N, 4, 2] -> [N, N]).

    Uses only standard ops (no data-dependent control flow), so it traces to ONNX
    with a dynamic N.
    """
    if x_factor != 1.0 or y_factor != 1.0:
        return quad_all_2_all_distance_reference(quads, x_factor, y_factor, allow_self_distance)

    quads = quads.float()
    d2 = _vertex_to_edge_sq_dist(quads)
    dist = torch.sqrt(torch.minimum(d2, d2.transpose(0, 1)))

    n = quads.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=quads.device)
    diag_val = 0.0 if allow_self_distance else float("inf")
    return torch.where(eye, torch.full_like(dist, diag_val), dist)


def ragged_quad_all_2_all_distance_v2_torch(
    embed_quads: torch.Tensor,
    region_counts: torch.Tensor,
    x_factor: float = 1.0,
    y_factor: float = 1.0,
    allow_self_distance: bool = True,
) -> torch.Tensor:
    """Drop-in replacement for the CUDA op.

    Args:
        embed_quads: [B, K, 4, 2] quads, padded per example.
        region_counts: [B] int64 count of valid quads per example.
    Returns:
        [B, K, K] distances, zero outside each example's valid block.
    """
    b, k = embed_quads.shape[0], embed_quads.shape[1]
    out = torch.zeros(b, k, k, dtype=embed_quads.dtype, device=embed_quads.device)
    for i, n in enumerate(region_counts.tolist()):
        n = int(n)
        if n > 0:
            out[i, :n, :n] = quad_all_2_all_distance(
                embed_quads[i, :n], x_factor, y_factor, allow_self_distance
            ).to(out.dtype)
    return out
