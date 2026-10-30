"""Traceable PyTorch modules that define the exported ONNX graphs.

Each wrapper reuses the upstream module (and its weights) unchanged and only
restates the parts of the upstream forward pass that the ONNX tracer can't follow
(Python loops over images, CUDA/C++ ops, data-dependent shapes) with standard
tensor ops. The math is the upstream math; see docs/ARCHITECTURE.md.

Graphs:
    detector    image [B,3,H,W]                       -> conf, rboxes, features
    sampler     features [B,C,h,w], image_index [1],
                quads [n,4,2]                         -> rec_input [n,C,8,T], rel_input [n,C,2,3]
    recognizer  rec_input [n,C,8,T]                   -> logits, features, token_ids, token_probs
    relational  rel_input, quads, rec_features (one image)
                                                      -> words, lines, line_log_var  [n, n+1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemotron_ocr.inference.models.quad_distance import quad_all_2_all_distance
from nemotron_ocr.inference.models.relational import NULL_CONNECTION_WEIGHT, get_directions

DETECTOR_DOWNSAMPLE = 4


class DetectorExport(nn.Module):
    """FOTSDetector without the unused QUAD-mode output."""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, image):
        conf, _, rboxes, features = self.detector(image)
        return conf, rboxes, features


class RecognizerExport(nn.Module):
    """TransformerRecognizer plus the argmax / winning-token probability that
    pipeline_v2 computes right after it, so a runtime can fetch only ids and probs."""

    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer

    def forward(self, rec_input):
        logits, features = self.recognizer(rec_input)
        token_ids = logits.argmax(dim=2)
        token_probs = torch.softmax(logits, dim=2).max(dim=2).values
        return logits, features, token_ids, token_probs


def _calc_quad_width(quads, output_height: int, max_width: int):
    """cpp/quad_rectify/quad_rectify_shared.h calc_quad_width with round_factor = 1."""
    q0, q1, q2, q3 = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    vw, vh, vh2 = q1 - q0, q3 - q0, q2 - q1
    quad_width = torch.sqrt((vw * vw).sum(-1))
    quad_height = (torch.sqrt((vh * vh).sum(-1)) + torch.sqrt((vh2 * vh2).sum(-1))) / 2.0
    quad_height = torch.clamp(quad_height, min=2.0)  # `if (h < 2) h = 2`
    quad_width = (output_height / quad_height) * quad_width
    quad_width = torch.clamp(torch.ceil(quad_width), min=1.0)  # max(rf, ceil(w / rf) * rf)
    quad_width = torch.clamp(quad_width, max=float(max_width))
    return torch.clamp(torch.floor(quad_width), min=2.0)


def quad_rectify_grid(quads, image_height, image_width, output_height: int, output_width: int,
                      isotropic: bool):
    """cpp/quad_rectify quad_rectify_forward (CPU path) as tensor ops.

    quads [n, 4, 2] in pixels -> grid [n, output_height, output_width, 2] in [-1, 1].
    Points past an isotropic quad's width map to (-1, -1) before normalisation,
    i.e. just outside [-1, 1].
    """
    dtype = quads.dtype
    q0 = quads[:, 0][:, None, None, :]
    a = (quads[:, 1] - quads[:, 0])[:, None, None, :]
    b = (quads[:, 3] - quads[:, 0])[:, None, None, :]
    c = (quads[:, 2] - quads[:, 1])[:, None, None, :]

    cols = torch.arange(output_width, dtype=dtype, device=quads.device)
    rows = torch.arange(output_height, dtype=dtype, device=quads.device)
    if isotropic:
        quad_width = _calc_quad_width(quads, output_height, output_width)[:, None]
    else:
        quad_width = torch.full((quads.shape[0], 1), float(output_width), dtype=dtype, device=quads.device)

    f_row = ((rows + 0.5) / float(output_height))[None, :, None, None]       # [1, H, 1, 1]
    f_col = ((cols[None, :] + 0.5) / quad_width)[:, None, :, None]           # [n, 1, W, 1]

    # Same evaluation order as the C++ (q0 + fCol*A + fRow*B + (fCol*fRow)*(C - B))
    pt = q0 + f_col * a + f_row * b + (f_col * f_row) * (c - b)
    inside = (cols[None, :] < quad_width)[:, None, :, None]
    pt = torch.where(inside, pt, torch.full_like(pt, -1.0))

    size = torch.stack((image_width, image_height)).to(dtype)
    return 2.0 * (pt / size) - 1.0


def indirect_grid_sample(features, grid):
    """cpp/better_grid_sample CPU forward for a single source image.

    features [1, C, h, w], grid [n, gh, gw, 2] -> [n, C, gh, gw]. Bilinear,
    align_corners semantics, clamped neighbours, 0 where the grid point is outside
    [-1, 1]. In-range points never put weight on an out-of-bounds neighbour, so
    GridSample's border padding reproduces the clamped fetch exactly.
    """
    n, gh, gw = grid.shape[0], grid.shape[1], grid.shape[2]
    flat = grid.reshape(1, n * gh, gw, 2)
    out = F.grid_sample(features, flat, mode="bilinear", padding_mode="border", align_corners=True)
    c = features.shape[1]
    out = out.reshape(c, n, gh, gw).permute(1, 0, 2, 3)
    inside = ((grid >= -1.0) & (grid <= 1.0)).all(dim=-1)  # [n, gh, gw]
    return out * inside[:, None].to(out.dtype)


class SamplerExport(nn.Module):
    """Quad rectification grids + feature sampling for the recognizer and relational inputs."""

    def __init__(self, rec_height: int, rec_width: int, rel_height: int = 2, rel_width: int = 3):
        super().__init__()
        self.rec_height, self.rec_width = rec_height, rec_width
        self.rel_height, self.rel_width = rel_height, rel_width

    def forward(self, features, image_index, quads):
        feats = torch.index_select(features, 0, image_index)  # [1, C, h, w]
        shape = torch._shape_as_tensor(feats)
        image_height = shape[2] * DETECTOR_DOWNSAMPLE
        image_width = shape[3] * DETECTOR_DOWNSAMPLE
        rec_grid = quad_rectify_grid(quads, image_height, image_width, self.rec_height, self.rec_width, True)
        rel_grid = quad_rectify_grid(quads, image_height, image_width, self.rel_height, self.rel_width, False)
        return indirect_grid_sample(feats, rec_grid), indirect_grid_sample(feats, rel_grid)


class RelationalExport(nn.Module):
    """GlobalRelationalModel.forward for one image, with static-shape neighbour selection.

    Differences in *form* from upstream (same values):

    - ``topk(k = min(n - 1, 15))`` becomes a fixed ``topk(15)`` over the distance
      matrix padded with +inf columns; slots past ``min(n - 1, 15)`` are masked and
      zeroed exactly like upstream's unfilled slots.
    - The scatter into the ``[n, n + 1]`` relation matrix sends masked slots to a
      spare column that is then dropped, so no index is written twice. Upstream sends
      them to column 0, where on CPU the last write (slot 15) wins when n <= 15; that
      value is copied into column 0 explicitly to match the CPU reference.
    """

    def __init__(self, relational):
        super().__init__()
        self.rel = relational
        self.z = relational.k - 1

    def _input_encoding(self, rel_input, quads, rec_features):
        rel = self.rel
        g_height, g_width = rel.quad_rectify_grid_size
        assert not rel.isotropic
        num_valid_pos = float(g_height * g_width)

        original_quads = quads / rel.quad_downscale
        mid_pts = original_quads.mean(dim=1)

        rq = rel.rect_proj(rel_input)
        avg = rq.flatten(2).sum(dim=2) / num_valid_pos
        rec = rel.recog_tx(rec_features).mean(dim=1)
        semantic = rel.combined_proj(torch.cat((avg, rec), dim=1))

        h1 = original_quads[:, 3] - original_quads[:, 0]
        h2 = original_quads[:, 2] - original_quads[:, 1]
        mp1 = original_quads[:, 0] + (h1 / 2)
        mp2 = original_quads[:, 1] + (h2 / 2)
        d1 = mp2 - mp1
        wdth = d1.norm(dim=1, keepdim=True)
        d1 = d1 / wdth.clamp_min(1e-6)
        hts = ((h1 + h2) / 2).norm(dim=1, keepdim=True)
        d2 = torch.stack([-d1[:, 1], d1[:, 0]], dim=-1)

        full = torch.cat((semantic, original_quads.flatten(1), d1, d2, wdth, hts), dim=1)
        return full, mid_pts, original_quads

    def forward(self, rel_input, quads, rec_features):
        z = self.z
        seq_len = z + 1
        rects, centers, quads_n = self._input_encoding(rel_input, quads.float(), rec_features.float())
        n = rects.shape[0]
        feat_dim = rects.shape[1]

        # n - 1 as a tensor, so the traced graph keeps it dynamic
        n_minus_1 = torch.arange(n, device=rects.device).max()
        z_i = torch.clamp(n_minus_1, max=z)

        dists = quad_all_2_all_distance(quads_n, allow_self_distance=False)
        pad = torch.full((n, z), float("inf"), dtype=dists.dtype, device=dists.device)
        topk_d, topk_idx = torch.topk(torch.cat((dists, pad), dim=1), k=z, dim=1, largest=False, sorted=True)

        slot = torch.arange(1, seq_len, device=rects.device)            # neighbour slots 1..z
        valid = (slot <= z_i)[None, :].expand(n, z)                      # [n, z]
        safe_idx = torch.where(valid, topk_idx, torch.zeros_like(topk_idx))

        nb_r = rects[safe_idx]                                           # [n, z, feat_dim]
        nb_c = centers[safe_idx]                                         # [n, z, 2]
        dirs = get_directions(quads_n, nb_c)                             # [n, z]
        from_r = rects[:, None, :].expand(n, z, feat_dim)
        nb = torch.cat((from_r, nb_r, topk_d[..., None], dirs[..., None]), dim=2)
        # Unfilled slots are 0 (where, not multiply: padded distances are +inf)
        nb = torch.where(valid[..., None], nb, torch.zeros_like(nb))

        first = torch.cat(
            (rects, torch.zeros_like(rects),
             torch.full((n, 1), -1.0, dtype=rects.dtype, device=rects.device),
             torch.full((n, 1), -2.0, dtype=rects.dtype, device=rects.device)),
            dim=1,
        )
        enc_input = torch.cat((first[:, None], nb), dim=1)               # [n, 16, 2*feat_dim + 2]
        mask = torch.cat((torch.zeros(n, 1, dtype=torch.bool, device=rects.device), ~valid), dim=1)

        out = self.rel.encoder[0](enc_input, src_key_padding_mask=mask)
        dots = self.rel.encoder[1](out)                                  # [n, 16, 3]

        # Column 0: slot 0, or (n <= 15) the last padded slot, as torch CPU scatter gives.
        col0_src = torch.where(z_i < z, dots[:, z], dots[:, 0])          # [n, 3]
        trash = n + 1
        nb_cols = torch.where(valid, topk_idx + 1, torch.full_like(topk_idx, trash))
        index = torch.cat((torch.zeros(n, 1, dtype=nb_cols.dtype, device=rects.device), nb_cols), dim=1)
        src = torch.cat((col0_src[:, None], dots[:, 1:]), dim=1)         # [n, 16, 3]

        rel_mat = torch.full((3, n, n + 2), NULL_CONNECTION_WEIGHT, dtype=dots.dtype, device=dots.device)
        rel_mat = torch.scatter(rel_mat, 2, index[None].expand(3, n, seq_len), src.permute(2, 0, 1))
        rel_mat = rel_mat[:, :, : n + 1]
        return rel_mat[0], rel_mat[1], rel_mat[2]
