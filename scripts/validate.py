#!/usr/bin/env python3
"""Validate the ONNX Runtime pipeline against the committed golden predictions.

For each variant, runs ``NemotronOCRONNX`` on every image in tests/golden/manifest.json
at word, sentence and paragraph level and compares with the PyTorch reference:

- text: every prediction's text must equal the reference, in the same order
- boxes: every box edge within --box-tol pixels (default 0.5) of the reference
- batch: all images in one call (paragraph level) must give the same results

Also reports pages/s (paragraph level, after a warm-up page) and peak process
memory (RSS). Prints a summary, writes a JSON report, and exits 1 on any mismatch.

Usage:
    scripts/validate.py                                   # both variants, providers=auto
    scripts/validate.py --variants english --providers cpu
    scripts/validate.py --providers CUDAExecutionProvider --report p40.json
"""

import argparse
import json
import logging
import platform
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import onnxruntime as ort  # noqa: E402

from nemotron_ocr_onnx import NemotronOCRONNX  # noqa: E402

MERGE_LEVELS = ("word", "sentence", "paragraph")


def peak_rss_mb():
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1e3  # bytes on macOS, KB on Linux
    except ImportError:  # Windows
        try:
            import psutil

            return psutil.Process().memory_info().peak_wset / 1e6
        except (ImportError, AttributeError):
            return None


def box_px(pred, hw):
    h, w = hw
    return [pred["left"] * w, pred["lower"] * h, pred["right"] * w, pred["upper"] * h]


def compare(preds, ref, hw):
    """Return (text_matches, total, max_box_err_px, first_problem)."""
    total = max(len(preds), len(ref))
    matches, max_err, problem = 0, 0.0, None
    if len(preds) != len(ref):
        problem = f"{len(preds)} predictions vs {len(ref)} in reference"
    for i, (p, r) in enumerate(zip(preds, ref)):
        if p["text"] == r["text"]:
            matches += 1
        elif problem is None:
            problem = f"#{i}: {p['text']!r} != {r['text']!r}"
        err = max(abs(a - b) for a, b in zip(box_px(p, hw), r["box_px"]))
        max_err = max(max_err, err)
    return matches, total, max_err, problem


def validate_variant(variant, args):
    golden_dir = args.golden / variant
    manifest = json.loads((args.golden / "manifest.json").read_text())
    keys = sorted(manifest["variants"][variant])
    refs = {k: json.loads((golden_dir / f"{k}.json").read_text()) for k in keys}
    images = {k: str(ROOT / refs[k]["image"]) for k in keys}
    missing = [images[k] for k in keys if not Path(images[k]).is_file()]
    if missing:
        raise SystemExit(f"missing test images: {missing} (run scripts/fetch_weights.sh for the sample)")

    t0 = time.perf_counter()
    ocr = NemotronOCRONNX(args.onnx_dir / variant, providers=args.providers,
                          intra_op_threads=args.threads)
    load_s = time.perf_counter() - t0

    ocr(images[keys[0]])  # warm-up (session initialisation, allocator growth)

    res = {"variant": variant, "session_providers": ocr.session_providers,
           "load_seconds": round(load_s, 2), "images": {}, "failures": []}
    matches = total = 0
    max_err = 0.0
    page_times = []
    for k in keys:
        hw = refs[k]["original_hw"]
        entry = {}
        for lvl in MERGE_LEVELS:
            t = time.perf_counter()
            preds = ocr(images[k], merge_level=lvl)
            dt = time.perf_counter() - t
            if lvl == "paragraph":
                page_times.append(dt)
            m, n, err, problem = compare(preds, refs[k]["predictions"][lvl], hw)
            matches, total, max_err = matches + m, total + n, max(max_err, err)
            entry[lvl] = {"text_match": f"{m}/{n}", "max_box_err_px": round(err, 4)}
            if problem or err > args.box_tol:
                res["failures"].append({"image": k, "level": lvl,
                                        "problem": problem or f"box error {err:.3f} px"})
        entry["seconds"] = round(page_times[-1], 3)
        res["images"][k] = entry
        print(f"  [{variant}] {k:14s} " + "  ".join(
            f"{lvl}={entry[lvl]['text_match']} ({entry[lvl]['max_box_err_px']:.3f}px)" for lvl in MERGE_LEVELS)
            + f"  {entry['seconds']:.2f}s", flush=True)

    res["peak_rss_mb"] = peak_rss_mb()  # single-page runs; the batch below needs more

    if not args.no_batch:
        t = time.perf_counter()
        batch = ocr([images[k] for k in keys], merge_level="paragraph")
        batch_s = time.perf_counter() - t
        bad = []
        for k, preds in zip(keys, batch):
            m, n, err, problem = compare(preds, refs[k]["predictions"]["paragraph"], refs[k]["original_hw"])
            if problem or err > args.box_tol:
                bad.append(k)
                res["failures"].append({"image": k, "level": "paragraph (batched)",
                                        "problem": problem or f"box error {err:.3f} px"})
        res["batch"] = {"images": len(keys), "seconds": round(batch_s, 2),
                        "pages_per_s": round(len(keys) / batch_s, 3), "mismatched": bad,
                        "peak_rss_mb": peak_rss_mb()}

    res["text_match_rate"] = matches / total if total else 1.0
    res["text_matches"] = f"{matches}/{total}"
    res["max_box_err_px"] = round(max_err, 4)
    res["pages_per_s"] = round(len(page_times) / sum(page_times), 3)
    res["mean_page_seconds"] = round(sum(page_times) / len(page_times), 3)
    res["passed"] = not res["failures"]
    return res


def main():
    ap = argparse.ArgumentParser(description="Validate the ONNX pipeline against the goldens.")
    ap.add_argument("--variants", nargs="+", default=["english", "multilingual"],
                    choices=["english", "multilingual"])
    ap.add_argument("--providers", nargs="+", default=["auto"],
                    help="'auto' or provider names/aliases (cpu, cuda, migraphx, coreml, dml)")
    ap.add_argument("--onnx-dir", type=Path, default=ROOT / "onnx")
    ap.add_argument("--golden", type=Path, default=ROOT / "tests" / "golden")
    ap.add_argument("--box-tol", type=float, default=0.5, help="max box edge error in pixels")
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = default)")
    ap.add_argument("--no-batch", action="store_true", help="skip the multi-image batch check")
    ap.add_argument("--report", type=Path, default=Path("validation_report.json"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.providers == ["auto"]:
        args.providers = "auto"

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "requested_providers": args.providers,
    }
    try:
        import torch

        env["torch"] = torch.__version__
    except ImportError:
        pass
    print(f"platform    {env['platform']} ({env['machine']})")
    print(f"python      {env['python']}")
    print(f"onnxruntime {env['onnxruntime']}  available: {', '.join(env['available_providers'])}")

    report = {"environment": env, "box_tol_px": args.box_tol, "variants": {}}
    for v in args.variants:
        print(f"\n== {v}")
        r = validate_variant(v, args)
        report["variants"][v] = r
        print("  providers: " + ", ".join(f"{g}={p}" for g, p in r["session_providers"].items()))
        print(f"  text match {r['text_matches']} ({100 * r['text_match_rate']:.2f}%), "
              f"max box error {r['max_box_err_px']:.3f} px, "
              f"{r['pages_per_s']:.2f} pages/s ({r['mean_page_seconds']:.2f} s/page), "
              f"peak RSS {r['peak_rss_mb'] or float('nan'):.0f} MB")
        if "batch" in r:
            print(f"  batch of {r['batch']['images']}: {r['batch']['pages_per_s']:.2f} pages/s, "
                  f"peak RSS {r['batch']['peak_rss_mb'] or float('nan'):.0f} MB, "
                  f"mismatched: {r['batch']['mismatched'] or 'none'}")
        for f in r["failures"]:
            print(f"  FAIL {f['image']} [{f['level']}]: {f['problem']}")
        print(f"  {'PASS' if r['passed'] else 'FAIL'}")

    report["passed"] = all(r["passed"] for r in report["variants"].values())
    args.report.write_text(json.dumps(report, indent=1))
    print(f"\n{'PASS' if report['passed'] else 'FAIL'}  (report: {args.report})")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
