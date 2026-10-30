# Nemotron OCR v2 pipeline: architecture map

This maps the upstream `nemotron-ocr` 2.0.2 inference pipeline
(`nemotron_ocr.inference.pipeline_v2.NemotronOCRV2`) as a basis for the ONNX port.
Paths are relative to the repository root; line numbers refer to the unmodified
upstream import (commit `9850078`).

Shapes below use the defaults: `infer_length = 1024`, one image, detector
downsample 4. `N` is the number of text regions surviving NMS, `B` the number of
images in a batch. The two variants differ only in recognizer size and charset
(from `model_config.json`):

| | v2_english | v2_multilingual |
|---|---|---|
| Detector backbone | RegNetX-8GF (ImageNet widths), FPN-style merge | same |
| Detector feature channels (`num_features`) | [1920, 512, 256, 128] | same |
| Recognizer conv input channels (`nic`) | 128 | 128 |
| Recognizer `depth` / d_model (`feature_depth`) | 128 / 256 | 256 / 512 |
| Recognizer layers / heads / FFN | 3 / 8 / 1024 | 6 / 8 / 2048 |
| `max_width` (sequence length T) | 32 | 128 |
| `num_tokens` (charset + 3 specials) | 858 (padded to 896 at load) | 14,247 (padded to 14,272) |
| Relational model | 4-layer pre-norm transformer, d=256, k=16 neighbours | same |

## 1. Call sequence

`NemotronOCRV2.__call__(images, merge_level)` → `_process_batch` (pipeline_v2.py:685):

| # | Phase | Code | Device in upstream | Notes |
|---|---|---|---|---|
| 0 | Construction | `NemotronOCR.__init__` → `_load_config`, `_load_charset`, `_load_models`, `_initialize_processors`; then `_pad_classifier_for_alignment(64)` | CUDA | Models moved with `.cuda()`. Classifier padded to a multiple of 64 outputs with bias `-inf` on the padding (argmax/softmax unaffected). `cudnn.benchmark = True`, TF32 enabled. |
| 1 | Preprocess | `_preprocess_batch` | CUDA | Decode to uint8 CHW on CPU (torchvision `read_image`/`decode_image`), copy to GPU, `.to(float16) / 255`, `pad_to_square(max(H, W), "bottom_right", value 1.0)`, `F.interpolate` bilinear `align_corners=True` to 1024×1024, right-pad width to a multiple of 128 with 0 (no-op at 1024). Output `[B, 3, 1024, 1024]` fp16. |
| 2 | Detector | `_run_detector_batched` | CUDA, autocast fp16 | Chunks of `detector_max_batch_size` (8). Returns conf, rboxes, features (see §2.1). |
| 2.5 | Prefilter | `_prefilter_detections` | CUDA | FCOS-style centerness from rbox TRBL distances × sigmoid(conf), 3×3 max-pool peak test, `> 0.5`; non-peaks set to logit −100. Pure torch. |
| 3 | NMS | `_run_nms` | CUDA | `sigmoid(conf)`; C++ `rrect_to_quads(rboxes, 4)` → `[B, 256, 256, 4, 2]`; C++ `quad_non_maximal_suppression(prob 0.5, iou 0.5, kernel 2×3, max_regions 0)` → `quads [N, 4, 2]`, `confidence [N]`, `region_counts [B]`. Returns early if N = 0. |
| 3b | Detector-only mode | `_process_batch` | | Scales quads to original pixels and returns boxes. |
| 4 | Rectify + sample | `_run_rectify_and_sample` | CUDA | `QuadRectify(8, max_width, isotropic=True)` (C++ `quad_rectify_forward`) → grid `[N, 8, T, 2]`; `QuadRectify(2, 3, isotropic=False)` → grid `[N, 2, 3, 2]`; C++ `region_counts_to_indices` → `[N]` image index; C++ `indirect_grid_sample_forward(det_features.float(), grid, idx, "bilinear")` twice → `rec_quads [N, 128, 8, T]`, `rel_quads [N, 128, 2, 3]` (fp32). |
| 5 | Recognizer | `_run_recognizer_chunked` | CUDA, autocast fp16 | Chunks of 128 regions, last chunk zero-padded; input `.half()`. Per chunk: `argmax` over tokens → ids `[N, T]`; `softmax(...).gather(ids)` → probs `[N, T]`; features kept `[N, T, D]`. Logits discarded per chunk. |
| 6 | Relational | `_run_relational_and_build_output` | CUDA, autocast fp16 | `relational(rel_quads, quads, region_counts.cpu(), rec_features)` → per image `words`, `lines`, `line_log_var_unc`, each `[n_i, n_i + 1]`. Then `softmax(words, dim=1)[:, 1:]`, quads scaled by `max(H, W) / 1024`, text confidence = geometric mean of per-token probs over tokens before the first EOS (id 1), excluding blanks (id 0). |
| 6b | Skip-relational mode | `_process_batch` | | Decodes text per region, no grouping. |
| 7 | Post-process | `_postprocess_batch` | CPU | `RecognitionTargetEncoder.convert_targets_to_labels` (C++ `decode_sequences`, token mapping from C++ `create_token_mapping`); `RelationalTargetEncoder.convert_targets_to_labels` (C++ `dense_relations_to_graph` for word→line and line→paragraph); prune invalid (`f_measure(det_conf, text_conf) > 0.5` and non-empty); build `TextBlock`s; `parse_relational_results(level=merge_level)`; `reorder_boxes(mode="top_left", dbscan_eps=10)`; normalise by original W/H. |

Output per image: list of `{"text", "confidence", "left", "upper", "right", "lower"}`
in [0, 1] image-relative units. Upstream's naming is inverted: `upper` is the
**max** y and `lower` the **min** y. The port keeps that format.

The two variants run the identical sequence; only tensor sizes differ (T = 32 vs
128, D = 256 vs 512, token count).

The older `NemotronOCR.__call__` path (pipeline.py:265, `_process_tensor`) is the
v1 single-image pipeline; `NemotronOCRV2` overrides `__call__` and does not use it.

## 2. Network boundaries

All three networks are plain `nn.Module`s in eval mode. Upstream runs them under
CUDA fp16 autocast; the port runs them in FP32.

### 2.1 Detector: `FOTSDetector` (models/detector/fots_detector.py)

| | Name | Shape | Dtype | Dynamic dims |
|---|---|---|---|---|
| in | `image` | `[B, 3, H, W]`, values in [0, 1] | fp32 (fp16 upstream) | B; H, W (multiples of 32; 1024×1024 by default) |
| out | `conf` (logits) | `[B, H/4, W/4]` | fp32 | B, H, W |
| out | `rboxes` | `[B, H/4, W/4, 5]`: top, right, bottom, left distances (`relu × scope`, scope 2048) + angle (`hardtanh × π`) | fp32 | B, H, W |
| out | `features` | `[B, 128, H/4, W/4]` (last merge level) | fp32 | B, H, W |

Normalisation (ImageNet mean/std, stored as fp16 buffers) is inside the model.
`forward` returns a 4-tuple `(conf, None, rboxes, features)` in RBOX mode; the
second element is the unused QUAD-mode output. The upsampling is `nearest` ×2, so
H and W must be divisible by 32.

### 2.2 Recognizer: `TransformerRecognizer` (models/recognizer.py)

| | Name | Shape | Dtype | Dynamic dims |
|---|---|---|---|---|
| in | `rec_input` | `[N, 128, 8, T]` | fp32 | N |
| out | `logits` | `[N, T, num_tokens]` | fp32 | N |
| out | `features` | `[N, T, D]` | fp32 | N |

Conv stack (3 × MaxPool(2, 1) reduces height 8 → 1) + learned position encoding
`[1, D, T]` + pre-norm `nn.TransformerEncoder` + final LayerNorm + Linear classifier.
T is fixed per variant (32 / 128) because `fixed_width` is true and the rectifier
always produces width `max_width`. Single forward pass, **not autoregressive**.
Token ids: 0 = blank, 1 = EOS, 2 = unused/special, 3.. = charset.

### 2.3 Relational: `GlobalRelationalModel` (models/relational.py)

Inputs: `rel_quads [N, 128, 2, 3]` fp32, `quads [N, 4, 2]` fp32 (detector-input
pixels), `region_counts [B]` int64 (CPU), `rec_features [N, T, D]`.
Outputs: dict of lists, one `[n_i, n_i + 1]` fp32 tensor per image for `words`,
`lines` and `line_log_var_unc`. Column 0 is the "null" (no connection) column;
non-neighbour entries are `-inf`.

The forward pass is not a pure network. It contains:

1. A per-region encoder (`rect_proj` 1×1 conv on the 2×3 sample, mean-pooled;
   `recog_tx` Linear on recognizer features, mean over T; `combined_proj` MLP with
   BatchNorm1d) plus geometric features (normalised quad corners, direction,
   width, height) → `[N, 128]`. Run in chunks of 128 (padding with copies).
2. **Per image** (Python loop): all-to-all quad distances via the C++ CUDA op
   `ragged_quad_all_2_all_distance_v2` (see Finding 2), `topk(k = min(n_i − 1, 15),
   largest=False)`, gather neighbour encodings and cosine directions → encoder input
   `[n_i, 16, 258]` plus a padding mask.
3. A 4-layer pre-norm `nn.TransformerEncoder` (d = 256, 8 heads) + Linear(256, 3) on
   `[n, 16, 256]`, chunked by 128.
4. **Per image**: `prohibit_self_connection` scatters the 16 outputs per row into a
   `[3, n_i, n_i + 1]` matrix of `-inf`.

Data-dependent parts: n_i (the number of regions), topk k, the scatter. Everything
operates within one image, so an ONNX graph for a single image with dynamic n is
exact; batches loop over images.

## 3. Calls into the C++ extension (`nemotron_ocr_cpp`)

Observed with `sys.setprofile` over a full v2 run at all three merge levels
(`text_region_grouping`, `calc_poly_min_rrect`, `get_poly_bounds_quad` and
`get_rel_continuation_cos` are bound but never reached on this path).

| Function | Stage | Caller | Tensors (upstream) | CPU implementation? |
|---|---|---|---|---|
| `rrect_to_quads` | 3 NMS | pipeline_v2.py `_run_nms` | CUDA | yes |
| `quad_non_maximal_suppression` | 3 NMS | pipeline_v2.py `_run_nms` | CUDA | yes (CPU impl copies inputs to CPU) |
| `region_counts_to_indices` | 4 | `_run_rectify_and_sample` | CUDA (NMS output) | device-agnostic (torch ops in a header) |
| `quad_rectify_forward` | 4 | post_processing/quad_rectify.py | CUDA | yes |
| `indirect_grid_sample_forward` | 4 | post_processing/indirect_grid_sample.py | CUDA | yes (OpenMP) |
| `ragged_quad_all_2_all_distance_v2` | 6 | models/relational.py `get_cdist` | CUDA | **no** (Finding 2) |
| `quad_rectify_calc_quad_width` | 6 | relational.py `get_input_encoding` (only if `isotropic`; it isn't) | — | yes |
| `create_token_mapping` | 0/7 | recognizer_encoder.py `_initialize` | CPU | CPU-only |
| `decode_sequences` | 7 | recognizer_encoder.py | CPU | CPU-only |
| `dense_relations_to_graph` | 7 | relational_encoder.py | CPU | CPU-only |

Backward functions (`quad_rectify_backward`, `indirect_grad_sample_backward`,
`rrect_to_quads_backward`) are training-only. Their CPU paths throw "not
implemented", which doesn't affect inference.

## 4. Hard-coded CUDA / fp16 assumptions in Python

| File:line (upstream) | Assumption |
|---|---|
| pipeline.py:31-32, pipeline_v2.py:48, :115 | Pad colour tensors created as fp16 and lazily moved with `.cuda()`. |
| pipeline.py:175, :232 | `torch.load(detector.pth / relational.pth)` without `map_location`. |
| pipeline.py:176, :217, :233 | `.cuda()` on detector, recognizer, relational. |
| pipeline.py:181, :222, :237 | `torch.backends.cudnn.benchmark = True`. |
| pipeline.py:331 | `convert_image_dtype(..., torch.float16)` (v1 path). |
| pipeline.py:349-351, :379, :393 | `PAD_COLOR.cuda()`, `quads.cuda()`, `input_indices.cuda()` (v1 path). |
| pipeline.py:359, :363, :408, :428 | `amp.autocast("cuda")` (v1 path). |
| pipeline.py:405-406 | Empty outputs created as fp16. |
| pipeline_v2.py:110-111 | TF32 enabled globally. |
| pipeline_v2.py:233-234, :253-254 | Pad colour `.cuda()`; image `.to("cuda")`, `.to(torch.float16)`. |
| pipeline_v2.py:338, :406, :486, :532, :543, :812 | `amp.autocast("cuda", enabled=True)` around every stage. |
| pipeline_v2.py:443 | `input_indices.cuda()`. |
| pipeline_v2.py:481, :489 | Recognizer input `.half()`; empty features fp16. |
| pipeline_v2.py:697-896 | `torch.cuda.synchronize()` for `verbose_post` timing. |
| pre_processing.py:53 | `interpolate_and_pad` moves images to `'cuda'` unless the pad colour is already on CUDA. |
| models/relational.py:13-16 | Imports the CUDA-only `ragged_quad_all_2_all_distance_v2`. |
| models/detector/fots_detector.py:296-300 | `input_mean` / `input_std` buffers are fp16 (harmless: promoted to fp32 against fp32 input; the fp16-rounded values are part of the checkpoint behaviour). |
| models/detector/fots_detector.py:31-32 | `get_prior_offsets` uses fp16 (QUAD mode only; unused in RBOX). |

Non-CUDA issues found while running on CPU (not in the original list):

- `models/detector/fots_detector.py:49`: `regnet_x_8gf(pretrained=True)` downloads
  the ImageNet RegNet weights from torch hub on every cold start, only to have them
  overwritten by `detector.pth`. `pretrained=True` also selects the ImageNet
  block widths (`w_0=80, group_width=120`) the checkpoint needs, so the flag can't
  simply be turned off.
- `cpp/non_maximal_suppression/non_maximal_suppression.cpp:238`: NMS outputs are
  allocated in pinned memory, which throws on machines without a CUDA driver.
- `cpp/scope_timer.h`, `non_maximal_suppression.cpp`: include `cuda_runtime.h` and
  call `cudaDeviceSynchronize()` from `.cpp` files, so even the CPU sources need the
  CUDA toolkit to compile.

## 5. Findings from the first read: confirmed / corrected

1. **The recognizer is not autoregressive: confirmed.** One forward pass returns
   `[N, T, num_tokens]` logits. pipeline_v2 takes per-position `argmax` and the
   winning token's softmax probability, then decodes with C++ `decode_sequences`
   via `RecognitionTargetEncoder` (no beam search / LM on this path). Exports as one
   graph.
2. **`ragged_quad_all_2_all_distance_v2` is CUDA-only and inside the relational
   forward: confirmed.** Called from `get_cdist` (per image, `allow_self_distance=False`,
   factors 1.0). `get_cdist_batched` is defined but unused. Math, from the kernel and
   `cpp/geometry.h`: for each (row, col) quad pair, min over the 4×4 edge pairs of
   `shortest_line_between_segments`, which only considers the 4 endpoint-to-segment
   candidates (crossing edges do **not** give 0), re-projected into the row quad's
   frame with positive components scaled by x/y factors. With factors 1.0 the
   re-projection is a rotation, so the result equals the symmetric minimum
   vertex-to-edge distance (up to float rounding).
3. **Networks joined through detector feature maps: confirmed.** Both the recognizer
   and relational inputs are bilinear samples (align_corners semantics, clamped
   neighbours, 0 outside [−1, 1]) of the detector's `[B, 128, 256, 256]` feature map
   at grids from `quad_rectify`. Sizes: recognizer grid 8 × T, isotropic (each quad's
   width in grid cells is `calc_quad_width`, rounded, capped at T; columns past it
   map to (−1, −1) → just outside the range → 0); relational grid 2 × 3, anisotropic.
   The feature map is 32 MB per page in fp32.
4. **Pipeline hard-wired to CUDA and FP16: confirmed**, plus the three non-CUDA issues
   in §4 (torch-hub download, pinned memory, CUDA headers in `.cpp` files).
5. **Other dispatchers have CPU branches: confirmed** for `rrect_to_quads`,
   `quad_rectify_forward`, NMS, `sparse_select` and `indirect_grid_sample_forward`.
   Only the backward passes lack CPU code (training-only).

### Additional finding: relational null-column collision

In `GlobalRelationalModel.forward`, `closest_flat` is initialised to 0 and slot 0 is
the null connection. When an image has n ≤ 15 regions, slots `n .. 15` are padding
(masked as keys) but keep index 0. `prohibit_self_connection` then does
`torch.scatter(..., index=closest)`, so column 0 receives writes from slot 0 **and**
every padding slot. On CPU the last write wins, so column 0 holds the output of slot
15 (a padded query position) instead of slot 0. On the sample page cropped to 10
regions: −10.78 (slot 15) instead of −10.25 (slot 0). On CUDA, duplicate-index
scatter is unordered, so upstream GPU results for small pages are nondeterministic
in this column. It only affects pages with ≤ 15 regions, and only the null logit
that enters the row softmax. The port reproduces the CPU (last-write-wins)
behaviour explicitly; it doesn't fix it, because that would change model output.
