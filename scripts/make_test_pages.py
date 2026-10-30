#!/usr/bin/env python3
"""Render the synthetic test pages used by the golden set (tests/images/*.png).

All text is original, written for this repository, and the pages are released
under the same terms as the code. Fonts: DejaVu, Liberation, FreeFont and
WenQuanYi Zen Hei (CJK) from Ubuntu 24.04's fonts-* packages. Rendered PNGs are
committed so every machine validates against identical pixels; rerunning this
script on another system may give slightly different rasterisation.

Usage: scripts/make_test_pages.py [--out tests/images]
"""

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = [Path("/usr/share/fonts/truetype"), Path("/usr/share/fonts/opentype")]


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for d in FONT_DIRS:
        hits = sorted(d.rglob(name))
        if hits:
            return ImageFont.truetype(str(hits[0]), size)
    raise FileNotFoundError(name)


SERIF = "DejaVuSerif.ttf"
SERIF_B = "DejaVuSerif-Bold.ttf"
SANS = "LiberationSans-Regular.ttf"
SANS_B = "LiberationSans-Bold.ttf"
MONO = "DejaVuSansMono.ttf"
FREE_SERIF_I = "FreeSerifItalic.ttf"
CJK = "wqy-zenhei.ttc"

PARAGRAPHS = [
    "The committee met on Tuesday morning to review the proposed changes to the "
    "village library opening hours. After a short discussion, members agreed to "
    "extend weekday hours until seven in the evening from the first of March.",
    "Volunteers will be needed to staff the front desk during the extra hours. "
    "Anyone interested should leave their name and a contact number with the "
    "librarian before the end of the month.",
    "The treasurer reported that the roof repairs came in under budget, leaving "
    "a small surplus that will be carried forward to next year's maintenance fund.",
    "Finally, the chair thanked everyone who helped with the autumn book sale, "
    "which raised more money than any previous year.",
]


def wrap(draw, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def page_letter():
    im = Image.new("RGB", (850, 1100), "white")
    d = ImageDraw.Draw(im)
    d.text((80, 70), "Riverside Community Library", font=font(SERIF_B, 30), fill="black")
    d.text((80, 115), "Minutes of the February committee meeting", font=font(FREE_SERIF_I, 20), fill="black")
    y, body = 180, font(SERIF, 17)
    for p in PARAGRAPHS:
        for line in wrap(d, p, body, 690):
            d.text((80, y), line, font=body, fill="black")
            y += 26
        y += 18
    d.text((80, y + 30), "Signed: J. Example, Secretary", font=body, fill="black")
    return im


def page_two_column():
    im = Image.new("RGB", (1000, 1300), "white")
    d = ImageDraw.Draw(im)
    d.text((60, 50), "Gardening Notes for Early Spring", font=font(SANS_B, 40), fill="black")
    body = font(SANS, 18)
    text = (
        "Start by clearing old leaves from borders and checking stakes and ties. "
        "Sow tomatoes and peppers indoors on a sunny windowsill, keeping the compost "
        "moist but not wet. Prune late flowering shrubs before new growth appears, "
        "and divide large clumps of perennials to create new plants for free. "
        "Lawns benefit from a light rake to remove moss, followed by a feed once the "
        "soil warms up. Watch out for slugs on young shoots after rain, and protect "
        "early blossom from frost with a sheet of fleece on cold nights. "
    ) * 2
    lines = wrap(d, text, body, 410)
    half = (len(lines) + 1) // 2
    for col, chunk in enumerate((lines[:half], lines[half:])):
        y = 140
        for line in chunk:
            d.text((60 + col * 470, y), line, font=body, fill="black")
            y += 27
    return im


def page_receipt():
    im = Image.new("RGB", (420, 1150), "white")
    d = ImageDraw.Draw(im)
    big, f = font(MONO, 26), font(MONO, 18)
    d.text((95, 40), "CORNER MARKET", font=big, fill="black")
    d.text((60, 85), "12 High Street, Anytown", font=f, fill="black")
    items = [
        ("Bread wholemeal", 2.15), ("Milk 2L", 1.65), ("Eggs x12", 3.40),
        ("Apples 1kg", 2.20), ("Coffee beans", 6.99), ("Pasta 500g", 0.95),
        ("Tomatoes tin", 0.70), ("Cheddar 400g", 3.75), ("Orange juice", 1.80),
        ("Rice 1kg", 1.49), ("Butter", 2.29), ("Bananas", 1.10),
    ]
    y = 150
    for name, price in items:
        d.text((30, y), name, font=f, fill="black")
        d.text((300, y), f"{price:6.2f}", font=f, fill="black")
        y += 34
    total = sum(p for _, p in items)
    d.line((30, y + 10, 390, y + 10), fill="black", width=2)
    d.text((30, y + 25), "TOTAL", font=big, fill="black")
    d.text((270, y + 25), f"{total:7.2f}", font=big, fill="black")
    d.text((30, y + 90), "CARD PAYMENT", font=f, fill="black")
    d.text((30, y + 124), "Thank you for shopping", font=f, fill="black")
    d.text((30, y + 158), "2024-03-14 10:42  #0071", font=f, fill="black")
    return im


def page_form():
    im = Image.new("RGB", (900, 1100), (250, 250, 247))
    d = ImageDraw.Draw(im)
    d.text((60, 50), "APPLICATION FOR A PARKING PERMIT", font=font(SANS_B, 30), fill="black")
    lab, val = font(SANS_B, 18), font(SERIF, 20)
    fields = [
        ("Surname", "Smith"), ("First name", "Alex"), ("House number", "42"),
        ("Street", "Maple Avenue"), ("Postcode", "AB1 2CD"),
        ("Vehicle registration", "XY12 ZZZ"), ("Make and model", "Blue hatchback"),
        ("Permit type", "Resident, annual"), ("Start date", "01/04/2024"),
    ]
    y = 140
    for k, v in fields:
        d.text((60, y), k, font=lab, fill=(40, 40, 40))
        d.rectangle((330, y - 6, 840, y + 30), outline=(120, 120, 120), width=1)
        d.text((340, y), v, font=val, fill=(10, 10, 90))
        y += 62
    d.text((60, y + 30), "Office use only", font=font(SANS, 14), fill=(90, 90, 90))
    return im


def page_rotated():
    im = Image.new("RGB", (1000, 1000), "white")
    f = font(SANS_B, 36)
    for i, (txt, ang, xy) in enumerate([
        ("Rotated by fifteen degrees", 15, (120, 260)),
        ("Upward slanting label", -12, (220, 560)),
        ("Vertical caption", 90, (850, 120)),
        ("Straight reference line", 0, (100, 820)),
    ]):
        tmp = Image.new("RGBA", (900, 90), (255, 255, 255, 0))
        ImageDraw.Draw(tmp).text((10, 10), txt, font=f, fill="black")
        x0, y0, x1, y1 = tmp.getbbox()
        tmp = tmp.crop((x0 - 8, y0 - 8, x1 + 8, y1 + 8))
        tmp = tmp.rotate(ang, expand=True, resample=Image.BICUBIC)
        im.paste(tmp, xy, tmp)
    return im


def page_sparse_small():
    # Fewer than 16 regions: exercises the relational model's small-page path.
    im = Image.new("RGB", (500, 260), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 50), "Please close", font=font(SANS_B, 40), fill="black")
    d.text((40, 120), "the gate behind you", font=font(SANS, 30), fill="black")
    d.text((40, 190), "Thank you", font=font(FREE_SERIF_I, 26), fill="black")
    return im


def page_slide():
    im = Image.new("RGB", (1280, 720), (24, 48, 96))
    d = ImageDraw.Draw(im)
    d.text((80, 60), "Quarterly Review", font=font(SANS_B, 64), fill="white")
    f = font(SANS, 34)
    for i, b in enumerate([
        "Orders up twelve percent on last quarter",
        "New warehouse opens in June",
        "Customer wait times down to two days",
        "Next steps: hire two more drivers",
    ]):
        d.ellipse((90, 215 + i * 90, 106, 231 + i * 90), fill=(255, 200, 60))
        d.text((130, 200 + i * 90), b, font=f, fill="white")
    return im


def page_low_contrast():
    im = Image.new("RGB", (900, 700), (236, 228, 205))
    d = ImageDraw.Draw(im)
    rnd = random.Random(7)
    for _ in range(4000):  # paper texture
        x, y = rnd.randrange(900), rnd.randrange(700)
        c = rnd.randrange(215, 235)
        d.point((x, y), fill=(c, c - 6, c - 25))
    f = font(SERIF, 24)
    y = 80
    for line in wrap(d, PARAGRAPHS[0] + " " + PARAGRAPHS[2], f, 760):
        d.text((70, y), line, font=f, fill=(120, 110, 95))
        y += 38
    return im


def page_dense():
    im = Image.new("RGB", (1100, 1400), "white")
    d = ImageDraw.Draw(im)
    f = font(SANS, 12)
    text = " ".join(PARAGRAPHS * 3)
    y = 40
    for line in wrap(d, text, f, 1000):
        d.text((50, y), line, font=f, fill="black")
        y += 17
    t = font(MONO, 12)
    for r in range(12):
        row = f"{r + 1:>3}  item-{r * 7 + 3:04d}  {r * 13.25:8.2f}  {(r * 37) % 101:>3}%  ok"
        d.text((50, y + 20 + r * 17), row, font=t, fill="black")
    return im


def page_multilingual():
    im = Image.new("RGB", (1000, 900), "white")
    d = ImageDraw.Draw(im)
    lat, cjk = font("DejaVuSans.ttf", 30), font(CJK, 34)
    lines = [
        (lat, "Bonjour, où est la gare ?"),
        (lat, "Grüße aus München und Köln"),
        (lat, "¿Dónde está la biblioteca?"),
        (lat, "Добрый день, как дела?"),
        (lat, "Καλημέρα από την Αθήνα"),
        (cjk, "今天天气很好"),
        (cjk, "東京駅はどこですか"),
        (cjk, "서울에서 만나요"),
    ]
    y = 60
    for fnt, txt in lines:
        d.text((70, y), txt, font=fnt, fill="black")
        y += 95
    return im


def page_blank():
    # No text at all: exercises the zero-detection path.
    im = Image.new("RGB", (640, 480), (252, 252, 252))
    d = ImageDraw.Draw(im)
    d.rectangle((40, 40, 600, 440), outline=(200, 200, 200), width=3)
    d.ellipse((260, 180, 380, 300), outline=(210, 210, 210), width=4)
    return im


PAGES = {
    "letter": page_letter,
    "two_column": page_two_column,
    "receipt": page_receipt,
    "form": page_form,
    "rotated": page_rotated,
    "sparse_small": page_sparse_small,
    "slide": page_slide,
    "low_contrast": page_low_contrast,
    "dense": page_dense,
    "multilingual": page_multilingual,
    "blank": page_blank,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "tests" / "images"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in PAGES.items():
        im = fn()
        path = out / f"{name}.png"
        im.save(path, optimize=True)
        print(f"{path}  {im.size[0]}x{im.size[1]}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
