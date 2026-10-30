#!/usr/bin/env python3
"""Export the Nemotron OCR v2 networks to ONNX and check them against the goldens.

Writes onnx/<variant>/{detector,sampler,recognizer,relational}.onnx (FP32, opset 17)
plus model_meta.json (charset, sizes) used by nemotron_ocr_onnx at runtime.

With --check (default) each graph is validated with onnx.checker, scanned for
non-standard ops, and run through ONNX Runtime (CPU) on every image in
tests/golden_full/<variant>/ (generate with scripts/make_golden.py --full). The
report (max abs diff and cosine similarity per output) is printed and saved to
onnx/<variant>/export_report.json.

Usage:
    scripts/export_onnx.py --variant english
    scripts/export_onnx.py --variant multilingual --no-check
"""

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nemotron_ocr.inference.pipeline import NemotronOCR  # noqa: E402
from nemotron_ocr_onnx.export_modules import (  # noqa: E402
    DetectorExport,
    RecognizerExport,
    RelationalExport,
    SamplerExport,
)

OPSET = 17
STANDARD_DOMAINS = {"", "ai.onnx"}


def export(module, args, path, input_names, output_names, dynamic_axes):
    module.eval()
    # The fused transformer fast path (aten::_transformer_encoder_layer_fwd) has no ONNX
    # symbolic; disabling it traces the same math as standard MatMul/Softmax/LayerNorm ops.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module,
            args,
            str(path),
            dynamo=False,
            opset_version=OPSET,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    torch.backends.mha.set_fastpath_enabled(fastpath)


def load_models(model_dir: Path):
    base = NemotronOCR(model_dir=str(model_dir), device="cpu", precision="fp32")
    for m in (base.detector, base.recognizer, base.relational):
        m.eval()
    return base


def export_all(variant: str, model_dir: Path, out: Path):
    base = load_models(model_dir)
    T = base.max_width
    c = base.detector.num_features[-1]
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    t0 = time.perf_counter()
    export(
        DetectorExport(base.detector),
        (torch.rand(1, 3, 1024, 1024),),
        out / "detector.onnx",
        ["image"], ["conf", "rboxes", "features"],
        {"image": {0: "batch", 2: "height", 3: "width"},
         "conf": {0: "batch", 1: "fh", 2: "fw"},
         "rboxes": {0: "batch", 1: "fh", 2: "fw"},
         "features": {0: "batch", 2: "fh", 3: "fw"}},
    )
    timings["detector"] = time.perf_counter() - t0

    g = torch.Generator().manual_seed(0)
    quads = torch.tensor([[[100.0, 100.0], [300.0, 110.0], [298.0, 150.0], [98.0, 140.0]]]).repeat(20, 1, 1)
    quads = quads + torch.rand(20, 1, 2, generator=g) * 600

    t0 = time.perf_counter()
    export(
        SamplerExport(8, T),
        (torch.rand(2, c, 256, 256), torch.tensor([1], dtype=torch.int64), quads),
        out / "sampler.onnx",
        ["features", "image_index", "quads"], ["rec_input", "rel_input"],
        {"features": {0: "batch", 2: "fh", 3: "fw"},
         "quads": {0: "regions"},
         "rec_input": {0: "regions"},
         "rel_input": {0: "regions"}},
    )
    timings["sampler"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    export(
        RecognizerExport(base.recognizer),
        (torch.rand(4, c, 8, T),),
        out / "recognizer.onnx",
        ["rec_input"], ["logits", "features", "token_ids", "token_probs"],
        {"rec_input": {0: "regions"}, "logits": {0: "regions"}, "features": {0: "regions"},
         "token_ids": {0: "regions"}, "token_probs": {0: "regions"}},
    )
    timings["recognizer"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    n = 20
    export(
        RelationalExport(base.relational),
        (torch.rand(n, c, 2, 3), quads[:n], torch.rand(n, T, base.recognizer.feature_depth)),
        out / "relational.onnx",
        ["rel_input", "quads", "rec_features"], ["words", "lines", "line_log_var"],
        {"rel_input": {0: "regions"}, "quads": {0: "regions"}, "rec_features": {0: "regions"},
         "words": {0: "regions", 1: "regions_plus_1"},
         "lines": {0: "regions", 1: "regions_plus_1"},
         "line_log_var": {0: "regions", 1: "regions_plus_1"}},
    )
    timings["relational"] = time.perf_counter() - t0

    meta = {
        "variant": variant,
        "opset": OPSET,
        "precision": "fp32",
        "infer_length": base.infer_length,
        "detector_downsample": 4,
        "max_width": T,
        "num_tokens": base.num_tokens,
        "feature_channels": c,
        "recognizer_feature_depth": base.recognizer.feature_depth,
        "relational_k": base.relational.k,
        "torch": torch.__version__,
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=1))
    for f in ("charset.txt", "model_config.json"):
        shutil.copyfile(model_dir / f, out / f)
    return timings


# ----------------------------------------------------------------------------- checks


def graph_report(path: Path):
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    domains = {n.domain for n in model.graph.node}
    ops = sorted({n.op_type for n in model.graph.node})
    nonstd = sorted(d for d in domains if d not in STANDARD_DOMAINS)
    return {
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "nodes": len(model.graph.node),
        "ops": ops,
        "non_standard_domains": nonstd,
    }


def compare(name, got, ref, results):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape:
        raise AssertionError(f"{name}: shape {got.shape} != {ref.shape}")
    fin = np.isfinite(ref)
    if not np.array_equal(np.isfinite(got), fin) or not np.array_equal(got[~fin], ref[~fin]):
        raise AssertionError(f"{name}: non-finite pattern differs")
    a, b = got[fin], ref[fin]
    max_abs = float(np.abs(a - b).max()) if a.size else 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(np.dot(a, b) / denom) if denom > 0 else 1.0
    r = results.setdefault(name, {"max_abs": 0.0, "min_cos": 1.0, "ref_max_abs": 0.0})
    r["max_abs"] = max(r["max_abs"], max_abs)
    r["min_cos"] = min(r["min_cos"], cos)
    r["ref_max_abs"] = max(r["ref_max_abs"], float(np.abs(b).max()) if b.size else 0.0)


def check(variant: str, out: Path, golden_dir: Path, threads: int):
    import onnxruntime as ort

    report = {"graphs": {}, "outputs": {}, "images": []}
    for g in ("detector", "sampler", "recognizer", "relational"):
        report["graphs"][g] = graph_report(out / f"{g}.onnx")

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = {g: ort.InferenceSession(str(out / f"{g}.onnx"), so, providers=["CPUExecutionProvider"])
            for g in report["graphs"]}

    files = sorted(golden_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no golden tensors in {golden_dir}; run scripts/make_golden.py --full")
    res = report["outputs"]
    token_mismatch = 0
    token_total = 0
    for f in files:
        g = np.load(f)
        conf, rboxes, feats = sess["detector"].run(None, {"image": g["det_input"]})
        compare("detector.conf", conf, g["det_conf"], res)
        compare("detector.rboxes", rboxes, g["det_rboxes"], res)
        compare("detector.features", feats, g["det_features"], res)
        entry = {"image": f.stem, "regions": int(g["nms_quads"].shape[0]) if "nms_quads" in g else 0}
        report["images"].append(entry)
        if "nms_quads" not in g:
            continue

        # Downstream graphs get golden inputs, so each is checked in isolation.
        rec_in, rel_in = sess["sampler"].run(None, {
            "features": g["det_features"], "image_index": np.array([0], dtype=np.int64),
            "quads": g["nms_quads"]})
        compare("sampler.rec_input", rec_in, g["rec_input"], res)
        compare("sampler.rel_input", rel_in, g["rel_input"], res)

        logits, rfeat, ids, probs = sess["recognizer"].run(None, {"rec_input": g["rec_input"]})
        k = g["rec_logits"].shape[0]
        compare("recognizer.logits", logits[:k], g["rec_logits"], res)
        compare("recognizer.features", rfeat, g["rec_features"], res)
        compare("recognizer.token_probs", probs, g["rec_probs"], res)
        mism = int((ids != g["rec_ids"]).sum())
        token_mismatch += mism
        token_total += ids.size
        entry["token_id_mismatches"] = mism

        words, lines, var = sess["relational"].run(None, {
            "rel_input": g["rel_input"], "quads": g["nms_quads"], "rec_features": g["rec_features"]})
        compare("relational.words", words, g["rel_words"], res)
        compare("relational.lines", lines, g["rel_lines"], res)
        compare("relational.line_log_var", var, g["rel_line_var"], res)

    report["token_ids"] = {"mismatches": token_mismatch, "total": token_total}
    return report


def print_report(variant, report):
    print(f"\n## {variant}\n")
    print("| graph | size MB | nodes | non-standard ops |")
    print("|---|---|---|---|")
    for g, r in report["graphs"].items():
        print(f"| {g} | {r['size_mb']} | {r['nodes']} | {', '.join(r['non_standard_domains']) or 'none'} |")
    print("\n| output | max abs diff | min cosine | max abs ref |")
    print("|---|---|---|---|")
    for k, r in report["outputs"].items():
        print(f"| {k} | {r['max_abs']:.2e} | {r['min_cos']:.9f} | {r['ref_max_abs']:.3g} |")
    t = report["token_ids"]
    print(f"\ntoken id mismatches: {t['mismatches']} / {t['total']}")


def main():
    ap = argparse.ArgumentParser(description="Export Nemotron OCR v2 networks to ONNX.")
    ap.add_argument("--variant", required=True, choices=["english", "multilingual"])
    ap.add_argument("--weights", type=Path, default=ROOT / "weights")
    ap.add_argument("--out", type=Path, default=None, help="default: onnx/<variant>")
    ap.add_argument("--golden-full", type=Path, default=ROOT / "tests" / "golden_full")
    ap.add_argument("--no-export", action="store_true", help="only run the checks")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    out = args.out or ROOT / "onnx" / args.variant
    if not args.no_export:
        timings = export_all(args.variant, args.weights / f"v2_{args.variant}", out)
        print("exported:", ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    if not args.no_check:
        report = check(args.variant, out, args.golden_full / args.variant, args.threads)
        (out / "export_report.json").write_text(json.dumps(report, indent=1))
        print_report(args.variant, report)


if __name__ == "__main__":
    main()
