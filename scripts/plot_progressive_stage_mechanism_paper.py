#!/usr/bin/env python3
"""Generate a paper-style stage-level mechanism figure."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/progressive_mechanism_20260428/stage_metrics.csv"
OUT_DIR = ROOT / "Paper/figures"
OUT_PNG = OUT_DIR / "progressive_stage_mechanism.png"

ORDER = [
    ("non_progressive", "fixed"),
    ("threshold_only_progressive", "early"),
    ("threshold_only_progressive", "mid"),
    ("threshold_only_progressive", "late"),
    ("safeearly_progressive", "early"),
    ("safeearly_progressive", "mid"),
    ("safeearly_progressive", "late"),
]

LABELS = {
    ("non_progressive", "fixed"): "Nonprog\nfixed\nrec/H1/0.35",
    ("threshold_only_progressive", "early"): "Threshold\nearly\nsafe/H1/0.90",
    ("threshold_only_progressive", "mid"): "Threshold\nmid\nrec/H1/0.35",
    ("threshold_only_progressive", "late"): "Threshold\nlate\nrec/H1/0.35",
    ("safeearly_progressive", "early"): "Safeearly\nearly\nsafe/H1/0.90",
    ("safeearly_progressive", "mid"): "Safeearly\nmid\nrec/H1/0.35",
    ("safeearly_progressive", "late"): "Safeearly\nlate\nrec/H2/0.55",
}

COLORS = {
    "non_progressive": (75, 102, 160),
    "threshold_only_progressive": (55, 132, 99),
    "safeearly_progressive": (190, 111, 62),
}

PANELS = [
    ("recursive_gate_rate_mean", "Recursive gate rate", "{:.2f}"),
    ("dead_end_rec_rate_mean", "Recursive dead-end rate", "{:.2f}"),
    ("perf_shield_time_ms_mean", "Shield time (ms)", "{:.0f}"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_rows() -> List[Dict[str, str]]:
    with INPUT.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_key = {
        (r["model"], r["progressive_stage"]): r
        for r in rows
        if r.get("row_type") == "aggregate" and r.get("split") == "eval"
    }
    return [by_key[key] for key in ORDER]


def val(row: Dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    return float(raw) if raw else 0.0


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill) -> None:
    x, y = xy
    width = draw.textlength(text, font=fnt)
    draw.text((x - width / 2, y), text, font=fnt, fill=fill)


def panel(draw: ImageDraw.ImageDraw, rows: List[Dict[str, str]], box, metric: str, title: str, fmt: str) -> None:
    x0, y0, x1, y1 = box
    text = (25, 30, 38)
    muted = (89, 97, 110)
    axis = (55, 63, 75)
    grid = (222, 227, 233)
    title_font = font(18, True)
    tick_font = font(11)
    label_font = font(10)

    draw.text((x0, y0), title, fill=text, font=title_font)
    top = y0 + 30
    bottom = y1 - 64
    left = x0 + 52
    right = x1 - 14
    values = [val(r, metric) for r in rows]
    max_v = max(values) if values else 1
    max_axis = max_v * 1.2 if max_v else 1

    for i in range(5):
        frac = i / 4
        y = bottom - frac * (bottom - top)
        draw.line((left, y, right, y), fill=grid, width=1)
        draw.text((x0 + 6, y - 7), fmt.format(max_axis * frac), fill=muted, font=tick_font)
    draw.line((left, top, left, bottom), fill=axis, width=2)
    draw.line((left, bottom, right, bottom), fill=axis, width=2)

    slot = (right - left) / len(rows)
    bar_w = min(42, slot * 0.55)
    for i, row in enumerate(rows):
        cx = left + slot * (i + 0.5)
        v = values[i]
        h = v / max_axis * (bottom - top) if max_axis else 0
        bx0, bx1 = cx - bar_w / 2, cx + bar_w / 2
        by0 = bottom - h
        draw.rectangle((bx0, by0, bx1, bottom), fill=COLORS[row["model"]])
        text_center(draw, (cx, by0 - 17), fmt.format(v), tick_font, text)

        lines = LABELS[(row["model"], row["progressive_stage"])].split("\n")
        for j, line in enumerate(lines):
            text_center(draw, (cx, bottom + 8 + j * 13), line, label_font, text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    scale = 2
    w, h = 900 * scale, 720 * scale
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font = font(18 * scale, True)
    subtitle_font = font(10 * scale)
    legend_font = font(10 * scale)
    text = (25, 30, 38)
    muted = (89, 97, 110)

    draw.text((30 * scale, 18 * scale), "Stage-level progressive mechanism", font=title_font, fill=text)
    draw.text(
        (30 * scale, 48 * scale),
        "Eval-stage aggregates; labels show mode / horizon / threshold.",
        font=subtitle_font,
        fill=muted,
    )
    legend_x, legend_y = 620 * scale, 20 * scale
    for idx, (model, label) in enumerate(
        [
            ("non_progressive", "non-progressive"),
            ("threshold_only_progressive", "threshold-only progressive"),
            ("safeearly_progressive", "safeearly progressive"),
        ]
    ):
        y = legend_y + idx * 18 * scale
        draw.rectangle((legend_x, y + 3 * scale, legend_x + 11 * scale, y + 14 * scale), fill=COLORS[model])
        draw.text((legend_x + 16 * scale, y), label, fill=text, font=legend_font)

    boxes = [
        (28 * scale, 82 * scale, 872 * scale, 265 * scale),
        (28 * scale, 290 * scale, 872 * scale, 473 * scale),
        (28 * scale, 498 * scale, 872 * scale, 690 * scale),
    ]
    for (metric, title, fmt), box in zip(PANELS, boxes):
        panel(draw, rows, box, metric, title, fmt)

    img.save(OUT_PNG, dpi=(300, 300))
    print(OUT_PNG)


if __name__ == "__main__":
    main()
