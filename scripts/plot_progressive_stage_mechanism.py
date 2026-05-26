#!/usr/bin/env python3
"""Generate a stage-level mechanism figure and table for progressive shielding."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/progressive_mechanism_20260428/stage_metrics.csv"
OUT_DIR = ROOT / "codex1_workspace"
OUT_PNG = OUT_DIR / "progressive_stage_mechanism.png"
OUT_MD = OUT_DIR / "progressive_stage_mechanism_table.md"

ORDER = [
    ("non_progressive", "fixed"),
    ("threshold_only_progressive", "early"),
    ("threshold_only_progressive", "mid"),
    ("threshold_only_progressive", "late"),
    ("safeearly_progressive", "early"),
    ("safeearly_progressive", "mid"),
    ("safeearly_progressive", "late"),
]

SHORT_MODEL = {
    "non_progressive": "nonprog",
    "threshold_only_progressive": "threshold",
    "safeearly_progressive": "safeearly",
}

METRICS = [
    ("recursive_gate_rate_mean", "recursive gate rate"),
    ("dead_end_rec_rate_mean", "dead-end rec rate"),
    ("perf_shield_time_ms_mean", "shield time (ms)"),
]


def _float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def load_rows() -> List[Dict[str, str]]:
    with INPUT.open(newline="") as f:
        all_rows = list(csv.DictReader(f))
    aggregates = {
        (row["model"], row["progressive_stage"]): row
        for row in all_rows
        if row.get("row_type") == "aggregate" and row.get("split") == "eval"
    }
    missing = [key for key in ORDER if key not in aggregates]
    if missing:
        raise RuntimeError(f"missing aggregate eval rows: {missing}")
    return [aggregates[key] for key in ORDER]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rows: List[Dict[str, str]],
    metric: str,
    title: str,
    box: tuple[int, int, int, int],
    y_label_fmt: str,
) -> None:
    x0, y0, x1, y1 = box
    axis_color = (70, 78, 89)
    grid_color = (218, 224, 230)
    text_color = (28, 34, 43)
    bar_colors = {
        "non_progressive": (94, 121, 177),
        "threshold_only_progressive": (67, 150, 116),
        "safeearly_progressive": (202, 126, 74),
    }
    title_font = font(20, bold=True)
    label_font = font(13)
    tick_font = font(12)

    draw.text((x0, y0), title, fill=text_color, font=title_font)
    plot_top = y0 + 36
    plot_bottom = y1 - 78
    plot_left = x0 + 54
    plot_right = x1 - 18
    values = [_float(row, metric) for row in rows]
    max_value = max(values) if values else 1.0
    max_axis = max_value * 1.18 if max_value > 0 else 1.0

    for i in range(5):
        frac = i / 4
        y = plot_bottom - frac * (plot_bottom - plot_top)
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)
        tick_value = max_axis * frac
        draw.text((x0 + 4, y - 8), y_label_fmt.format(tick_value), fill=(88, 96, 108), font=tick_font)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)

    slot = (plot_right - plot_left) / len(rows)
    bar_width = min(46, slot * 0.58)
    for i, row in enumerate(rows):
        value = values[i]
        cx = plot_left + slot * (i + 0.5)
        bar_h = 0 if max_axis == 0 else value / max_axis * (plot_bottom - plot_top)
        bx0 = cx - bar_width / 2
        bx1 = cx + bar_width / 2
        by0 = plot_bottom - bar_h
        color = bar_colors[row["model"]]
        draw.rounded_rectangle((bx0, by0, bx1, plot_bottom), radius=3, fill=color)
        value_text = y_label_fmt.format(value)
        tw = draw.textlength(value_text, font=tick_font)
        draw.text((cx - tw / 2, by0 - 18), value_text, fill=text_color, font=tick_font)

        label = f"{SHORT_MODEL[row['model']]}\n{row['progressive_stage']}"
        lines = label.split("\n")
        for j, line in enumerate(lines):
            tw = draw.textlength(line, font=label_font)
            draw.text((cx - tw / 2, plot_bottom + 8 + j * 16), line, fill=text_color, font=label_font)


def draw_figure(rows: List[Dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1600, 980
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = font(30, bold=True)
    sub_font = font(16)
    note_font = font(15)
    text_color = (26, 32, 41)
    muted = (86, 94, 105)

    draw.text((50, 34), "Progressive Stage-Level Mechanism", fill=text_color, font=title_font)
    draw.text(
        (50, 76),
        "Eval-stage aggregates from runs/progressive_mechanism_20260428/stage_metrics.csv",
        fill=muted,
        font=sub_font,
    )

    boxes = [(45, 128, 1545, 382), (45, 402, 1545, 656), (45, 676, 1545, 930)]
    fmts = ["{:.2f}", "{:.2f}", "{:.0f}"]
    for (metric, title), box, fmt in zip(METRICS, boxes, fmts):
        draw_panel(draw, rows, metric, title, box, fmt)

    legend_x = 1050
    legend_y = 35
    legend = [
        ((94, 121, 177), "non_progressive"),
        ((67, 150, 116), "threshold_only_progressive"),
        ((202, 126, 74), "safeearly_progressive"),
    ]
    for i, (color, label) in enumerate(legend):
        y = legend_y + i * 24
        draw.rectangle((legend_x, y + 3, legend_x + 18, y + 18), fill=color)
        draw.text((legend_x + 26, y), label, fill=text_color, font=note_font)

    note = "Mode/horizon/threshold: nonprog fixed rec/H1/0.35; threshold early safe/H1/0.90 then rec/H1/0.35; safeearly late rec/H2/0.55."
    draw.text((50, 948), note, fill=muted, font=note_font)
    image.save(OUT_PNG)


def write_markdown(rows: List[Dict[str, str]]) -> None:
    lines = [
        "# Progressive Stage Mechanism Table",
        "",
        "数据来源：`runs/progressive_mechanism_20260428/stage_metrics.csv`，仅使用 `row_type=aggregate` 且 `split=eval` 的行。",
        "",
        "| model | stage | shield mode | horizon | threshold | recursive_gate_rate | dead_end_rec_rate | perf_shield_time_ms | perf_recursive_time_ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| `{model}` | {stage} | {mode} | {horizon:.0f} | {threshold:.2f} | {gate:.4f} | {dead:.4f} | {shield:.2f} | {rec:.2f} |".format(
                model=row["model"],
                stage=row["progressive_stage"],
                mode=row["effective_shield_mode"],
                horizon=_float(row, "effective_lookahead_horizon_mean"),
                threshold=_float(row, "effective_risk_threshold_mean"),
                gate=_float(row, "recursive_gate_rate_mean"),
                dead=_float(row, "dead_end_rec_rate_mean"),
                shield=_float(row, "perf_shield_time_ms_mean"),
                rec=_float(row, "perf_recursive_time_ms_mean"),
            )
        )
    lines.extend(
        [
            "",
            "可支撑表述：early 阶段主要停留在 safe / hard-safe 层，`threshold_only_progressive` 在 mid/late 切入 H=1 recursive layer；`safeearly_progressive` late 切入 H=2，但这不应被写成 final learned policy 稳定更优。",
            "",
            "不能写太满：该表说明 stage 机制差异和 runtime/gate 分布，不单独证明 threshold-only 的收益完全来自某个唯一因果机制。",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    draw_figure(rows)
    write_markdown(rows)
    print(OUT_PNG)
    print(OUT_MD)


if __name__ == "__main__":
    main()
