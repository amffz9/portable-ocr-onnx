#!/usr/bin/env python3
"""Run OCR on images or PDFs with the ONNX Runtime pipeline.

Prints the recognised text (one region per line, in reading order) or writes JSON
with text, confidence and pixel boxes. PDFs are rendered with Poppler's pdftoppm
(install poppler-utils / `brew install poppler`).

Usage:
    scripts/ocr.py page.png
    scripts/ocr.py scan.pdf --variant multilingual --level sentence
    scripts/ocr.py *.png --json results.json --min-confidence 0.9
    scripts/ocr.py doc.pdf --providers cuda
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from nemotron_ocr_onnx import NemotronOCRONNX  # noqa: E402


def render_pdf(pdf: Path, dpi: int, out_dir: Path):
    if shutil.which("pdftoppm") is None:
        raise SystemExit("PDF input needs pdftoppm (poppler-utils / `brew install poppler`)")
    prefix = out_dir / pdf.stem
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(prefix)], check=True)
    # pdftoppm zero-pads page numbers to the page count's width, so sort numerically
    pages = sorted(out_dir.glob(f"{pdf.stem}-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    return [(f"{pdf.name} page {i}", p) for i, p in enumerate(pages, 1)]


def main():
    ap = argparse.ArgumentParser(description="OCR images or PDFs with the ONNX pipeline.")
    ap.add_argument("inputs", nargs="+", type=Path, help="image files (png, jpg, ...) or PDFs")
    ap.add_argument("--variant", default="english", choices=["english", "multilingual"])
    ap.add_argument("--model-dir", type=Path, default=None, help="default: onnx/<variant>")
    ap.add_argument("--level", default="paragraph", choices=["word", "sentence", "paragraph"],
                    help="word = single words, sentence = lines, paragraph = blocks (default)")
    ap.add_argument("--providers", nargs="+", default=["auto"],
                    help="'auto' or provider names/aliases (cpu, cuda, migraphx, coreml, dml)")
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = default)")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop regions below this confidence (e.g. 0.9 to skip most gibberish)")
    ap.add_argument("--dpi", type=int, default=150, help="PDF render resolution")
    ap.add_argument("--json", type=Path, default=None, help="write results to this JSON file")
    ap.add_argument("-v", "--verbose", action="store_true", help="log which provider each graph uses")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    providers = "auto" if args.providers == ["auto"] else args.providers
    ocr = NemotronOCRONNX(args.model_dir or ROOT / "onnx" / args.variant,
                          providers=providers, intra_op_threads=args.threads)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        pages = []
        for path in args.inputs:
            if not path.is_file():
                raise SystemExit(f"not found: {path}")
            if path.suffix.lower() == ".pdf":
                pages += render_pdf(path, args.dpi, Path(tmp))
            else:
                pages.append((path.name, path))

        for name, page in pages:
            w, h = Image.open(page).size
            regions = []
            for p in ocr(str(page), merge_level=args.level):
                if p["confidence"] < args.min_confidence:
                    continue
                # "upper" is the larger y in image coordinates (upstream naming)
                regions.append({
                    "text": p["text"],
                    "confidence": round(p["confidence"], 4),
                    "box_px": [round(p["left"] * w, 1), round(p["lower"] * h, 1),
                               round(p["right"] * w, 1), round(p["upper"] * h, 1)],
                })
            results.append({"page": name, "width": w, "height": h, "regions": regions})
            if not args.json:
                print(f"=== {name}")
                for r in regions:
                    print(r["text"])
                print()

    if args.json:
        args.json.write_text(json.dumps({"variant": args.variant, "level": args.level,
                                         "providers": ocr.session_providers, "pages": results},
                                        indent=1, ensure_ascii=False))
        print(f"wrote {args.json} ({len(results)} pages)")


if __name__ == "__main__":
    main()
