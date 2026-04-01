from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_TAGS = [
    "train/episode_reward",
    "train/loss",
    "train/search_rate",
    "train/coverage_rate",
    "eval/search_rate",
    "eval/avg_reward",
]


def load_scalar_series(log_dir: str, tags: Sequence[str] | None = None) -> Dict[str, List[Tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as e:
        raise RuntimeError("tensorboard is required for visualization. Please install it first.") from e

    tags = list(tags or DEFAULT_TAGS)
    acc = event_accumulator.EventAccumulator(str(log_dir), size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload()
    avail = set(acc.Tags().get("scalars", []))

    out: Dict[str, List[Tuple[int, float]]] = {}
    for tag in tags:
        if tag not in avail:
            continue
        events = acc.Scalars(tag)
        by_step: Dict[int, Tuple[float, float]] = {}
        for e in events:
            step = int(e.step)
            wall_time = float(e.wall_time)
            value = float(e.value)
            prev = by_step.get(step)
            if prev is None or wall_time >= prev[0]:
                by_step[step] = (wall_time, value)
        out[tag] = [(step, by_step[step][1]) for step in sorted(by_step.keys())]
    return out


def _polyline_points(points: Iterable[Tuple[int, float]], width: int, height: int, pad: int) -> Tuple[str, Dict[str, float]]:
    pts = list(points)
    if not pts:
        return "", {}

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    if max_x == min_x:
        max_x = min_x + 1
    if max_y == min_y:
        max_y = min_y + 1e-6

    chart_w = width - 2 * pad
    chart_h = height - 2 * pad
    coords = []
    for x, y in pts:
        px = pad + (x - min_x) * chart_w / (max_x - min_x)
        py = height - pad - (y - min_y) * chart_h / (max_y - min_y)
        coords.append(f"{px:.2f},{py:.2f}")

    meta = {
        "min_x": float(min(xs)),
        "max_x": float(max(xs)),
        "min_y": float(min(ys)),
        "max_y": float(max(ys)),
        "last_y": float(ys[-1]),
    }
    return " ".join(coords), meta


def _build_svg(points: Sequence[Tuple[int, float]], width: int = 860, height: int = 240, pad: int = 34) -> Tuple[str, Dict[str, float]]:
    poly, meta = _polyline_points(points, width=width, height=height, pad=pad)
    if not poly:
        return "", {}

    lines = []
    for i in range(5):
        y = pad + i * (height - 2 * pad) / 4
        lines.append(
            f'<line x1="{pad}" y1="{y:.2f}" x2="{width-pad}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        f'{"".join(lines)}'
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2.2" points="{poly}"/>'
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#9ca3af" stroke-width="1"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#9ca3af" stroke-width="1"/>'
        "</svg>"
    )
    return svg, meta


def generate_training_report(
    log_dir: str,
    output_html: str = "runs/hrvdn/report.html",
    tags: Sequence[str] | None = None,
    eval_metrics: Dict[str, float] | None = None,
) -> Path:
    log_path = Path(log_dir)
    series = load_scalar_series(str(log_path), tags=tags)

    cards = []
    for tag, points in series.items():
        if not points:
            continue
        svg, meta = _build_svg(points)
        cards.append(
            (
                f"<section class='card'>"
                f"<h3>{tag}</h3>"
                f"{svg}"
                f"<p class='meta'>step {int(meta['min_x'])} -> {int(meta['max_x'])}, "
                f"min {meta['min_y']:.4f}, max {meta['max_y']:.4f}, last {meta['last_y']:.4f}</p>"
                "</section>"
            )
        )

    if eval_metrics:
        rows = "".join(
            f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" for k, v in eval_metrics.items()
        )
        eval_block = (
            "<section class='card'>"
            "<h3>checkpoint eval metrics</h3>"
            "<table><thead><tr><th>metric</th><th>value</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "</section>"
        )
    else:
        eval_block = ""

    if not cards:
        cards_html = "<p class='empty'>No scalar data found. Please check the log directory.</p>"
    else:
        cards_html = "".join(cards)

    html = (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'><title>HRVDN Training Report</title>"
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;background:#f8fafc;color:#0f172a;margin:0;padding:24px;}"
        "h1{margin:0 0 16px 0;}"
        ".hint{color:#475569;margin:0 0 18px 0;}"
        ".card{background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 14px 10px 14px;"
        "box-shadow:0 1px 2px rgba(15,23,42,.04);margin-bottom:14px;overflow-x:auto;}"
        ".card h3{margin:0 0 8px 0;font-size:16px;}"
        ".meta{margin:8px 0 0 0;font-size:13px;color:#475569;}"
        ".empty{padding:12px;background:#fff;border:1px dashed #cbd5e1;border-radius:8px;}"
        "table{width:100%;border-collapse:collapse;}"
        "th,td{border:1px solid #e2e8f0;padding:8px 10px;text-align:left;}"
        "th{background:#f1f5f9;}"
        "</style></head><body>"
        "<h1>HRVDN Training Report</h1>"
        f"<p class='hint'>logdir: {log_path}</p>"
        f"{eval_block}"
        f"{cards_html}"
        "</body></html>"
    )

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def parse_tags(raw: str | None) -> List[str] | None:
    if not raw:
        return None
    tags = [x.strip() for x in raw.split(",") if x.strip()]
    return tags or None
