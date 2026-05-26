from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


MODEL_LABELS = {
    "recursive_risk": "recursive(risk_base)",
    "risk_base": "recursive + risk_base",
    "risk_base_normdpm": "recursive + risk_base (normdpm)",
    "recursive_full": "recursive(full)",
    "safe": "safe",
    "off": "off",
    "safe_sequential": "safe + sequential",
    "safe_exact": "safe + exact",
    "safe_rescue": "safe + rescue",
    "recursive_full_sequential": "recursive(full) + sequential",
    "recursive_full_exact": "recursive(full) + exact",
    "recursive_full_rescue": "recursive(full) + rescue",
    "recursive_risk_sequential": "recursive(risk) + sequential",
    "recursive_risk_exact": "recursive(risk) + exact",
    "recursive_risk_rescue": "recursive(risk) + rescue",
    "recursive_risk_progressive_rescue": "recursive(risk) + progressive + rescue",
    "recursive_full_rescue_h1": "recursive(full) + rescue + H=1",
    "recursive_full_rescue_h2": "recursive(full) + rescue + H=2",
    "recursive_risk_rescue_h1": "recursive(risk) + rescue + H=1",
    "recursive_risk_rescue_h2": "recursive(risk) + rescue + H=2",
    "recursive_risk_rescue_h2_eta55_ref": "recursive(risk) + rescue + H=2 + eta=0.55",
    "recursive_risk_rescue_h1_eta35": "recursive(risk) + rescue + H=1 + eta=0.35",
    "recursive_risk_rescue_h1_eta45": "recursive(risk) + rescue + H=1 + eta=0.45",
    "recursive_risk_rescue_h1_eta55": "recursive(risk) + rescue + H=1 + eta=0.55",
    "recursive_risk_rescue_h1_eta65": "recursive(risk) + rescue + H=1 + eta=0.65",
    "recursive_risk_rescue_h1_eta75": "recursive(risk) + rescue + H=1 + eta=0.75",
    "recursive_risk_rescue_h1_eta85": "recursive(risk) + rescue + H=1 + eta=0.85",
    "safe_rescue_h1": "safe + rescue + H=1",
    "safe_rescue_h2": "safe + rescue + H=2",
    "h1_ckpt_h1_shield": "H1 ckpt + H1 shield",
    "h1_ckpt_h2_shield": "H1 ckpt + H2 shield",
    "h2_ckpt_h1_shield": "H2 ckpt + H1 shield",
    "h2_ckpt_h2_shield": "H2 ckpt + H2 shield",
    "non_progressive": "Non-progressive",
    "threshold_only_progressive": "Threshold-only progressive",
    "safeearly_progressive": "Safe-early progressive",
    "threshold_only_dual_progressive": "Threshold-only + dual scheduling",
}

MODEL_COLORS = {
    "recursive_risk": "#1d4ed8",
    "risk_base": "#1d4ed8",
    "risk_base_normdpm": "#1d4ed8",
    "recursive_full": "#7c3aed",
    "safe": "#0f766e",
    "off": "#b45309",
    "safe_sequential": "#0f766e",
    "safe_exact": "#0ea5e9",
    "safe_rescue": "#16a34a",
    "recursive_full_sequential": "#7c3aed",
    "recursive_full_exact": "#a855f7",
    "recursive_full_rescue": "#6d28d9",
    "recursive_risk_sequential": "#1d4ed8",
    "recursive_risk_exact": "#2563eb",
    "recursive_risk_rescue": "#0284c7",
    "recursive_risk_progressive_rescue": "#e11d48",
    "recursive_full_rescue_h1": "#7c3aed",
    "recursive_full_rescue_h2": "#4c1d95",
    "recursive_risk_rescue_h1": "#0284c7",
    "recursive_risk_rescue_h2": "#075985",
    "recursive_risk_rescue_h2_eta55_ref": "#0f766e",
    "recursive_risk_rescue_h1_eta35": "#1d4ed8",
    "recursive_risk_rescue_h1_eta45": "#2563eb",
    "recursive_risk_rescue_h1_eta55": "#3b82f6",
    "recursive_risk_rescue_h1_eta65": "#60a5fa",
    "recursive_risk_rescue_h1_eta75": "#93c5fd",
    "recursive_risk_rescue_h1_eta85": "#bfdbfe",
    "safe_rescue_h1": "#16a34a",
    "safe_rescue_h2": "#166534",
    "h1_ckpt_h1_shield": "#0284c7",
    "h1_ckpt_h2_shield": "#0f766e",
    "h2_ckpt_h1_shield": "#7c3aed",
    "h2_ckpt_h2_shield": "#b45309",
    "non_progressive": "#1d4ed8",
    "threshold_only_progressive": "#f97316",
    "safeearly_progressive": "#16a34a",
    "threshold_only_dual_progressive": "#dc2626",
}

TASK_METRICS = [
    {
        "key": "search_rate",
        "label": "Search Rate",
        "unit": "ratio",
        "direction": "higher",
        "description": "Episode-level target discovery rate.",
    },
    {
        "key": "coverage_ratio",
        "label": "Coverage Ratio",
        "unit": "ratio",
        "direction": "higher",
        "description": "Mean map coverage ratio at episode end.",
    },
    {
        "key": "error_rate",
        "label": "Belief Error",
        "unit": "ratio",
        "direction": "lower",
        "description": "Mean target belief-map error.",
    },
]

SAFETY_METRICS = [
    {
        "key": "collision_count",
        "label": "Collision Count",
        "unit": "count / episode",
        "direction": "lower",
        "description": "Average collisions per episode.",
    },
    {
        "key": "near_miss_rate",
        "label": "Near-Miss Rate",
        "unit": "ratio",
        "direction": "lower",
        "description": "Fraction of steps with near-miss events.",
    },
    {
        "key": "guarantee_broken_rate",
        "label": "Guarantee-Broken Rate",
        "unit": "ratio",
        "direction": "lower",
        "description": "Fraction of steps where A_hard was empty and the hard-safety guarantee was explicitly broken by dead-end semantics.",
    },
]

SHIELD_METRICS = [
    {
        "key": "shield_trigger_rate",
        "label": "Shield Trigger Rate",
        "unit": "ratio",
        "direction": "lower",
        "description": "Fraction of steps where shield intervened.",
    },
    {
        "key": "action_replacement_rate",
        "label": "Action Replacement Rate",
        "unit": "ratio",
        "direction": "lower",
        "description": "Fraction of steps where actor had to reselect.",
    },
    {
        "key": "recursive_gate_rate",
        "label": "Recursive Gate Rate",
        "unit": "ratio",
        "direction": "lower",
        "description": "Fraction of agent-steps upgraded to A_rec.",
    },
    {
        "key": "avg_hard_action_count",
        "label": "Avg |A_hard|",
        "unit": "count",
        "direction": "higher",
        "description": "Average cheap always-on hard-safe action-set size.",
    },
    {
        "key": "avg_rec_action_count",
        "label": "Avg |A_rec|",
        "unit": "count",
        "direction": "higher",
        "description": "Average recursive action-set size after upgrade.",
    },
    {
        "key": "dead_end_hard_rate",
        "label": "Dead-End A_hard Rate",
        "unit": "ratio",
        "direction": "lower",
        "description": "Fraction of steps where A_hard is empty.",
    },
    {
        "key": "dead_end_rec_rate",
        "label": "Dead-End A_rec Rate",
        "unit": "ratio",
        "direction": "lower",
        "description": "Fraction of steps where A_rec is empty.",
    },
    {
        "key": "exact_hard_false_empty_rate",
        "label": "Seq Empty -> Exact Nonempty",
        "unit": "ratio",
        "direction": "lower",
        "description": "Conditional false-empty rate: among sequential-empty agent-queries, how often exact A_hard is non-empty.",
    },
    {
        "key": "seq_nonempty_exact_empty_rate",
        "label": "Seq Nonempty -> Exact Empty",
        "unit": "ratio",
        "direction": "lower",
        "description": "Conditional false-positive rate: among sequential-nonempty agent-queries, how often exact A_hard is empty.",
    },
    {
        "key": "seq_exact_jaccard",
        "label": "Seq/Exact Jaccard",
        "unit": "ratio",
        "direction": "higher",
        "description": "Mean agent-level Jaccard similarity between sequential A_hard and exact A_hard.",
    },
    {
        "key": "perf_exact_hard_time_ms",
        "label": "Exact Hard Time",
        "unit": "ms / step",
        "direction": "lower",
        "description": "Average exact A_hard oracle time per environment step.",
    },
    {
        "key": "perf_recursive_time_ms",
        "label": "Stronger Check Time",
        "unit": "ms / step",
        "direction": "lower",
        "description": "Average recursive / look-ahead stronger-check time per environment step.",
    },
    {
        "key": "perf_recursive_work_time_ms",
        "label": "Stronger Work Time",
        "unit": "ms / step",
        "direction": "lower",
        "description": "Cumulative recursive helper work per environment step. Unlike Stronger Check Time, this may exceed wall-clock because nested calls are accumulated.",
    },
    {
        "key": "perf_recursive_candidate_checks",
        "label": "Stronger Candidate Checks",
        "unit": "count / step",
        "direction": "lower",
        "description": "Average number of current-step candidate actions evaluated by the stronger recursive / look-ahead layer.",
    },
]

RETURN_METRICS = [
    {
        "key": "episode_return",
        "label": "Episode Return",
        "unit": "return",
        "direction": "higher",
        "description": "Average episode return under aligned reward normalization.",
    },
    {
        "key": "avg_reward",
        "label": "Average Reward",
        "unit": "return",
        "direction": "higher",
        "description": "Mean per-episode reward under the current evaluation setup.",
    },
]


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_scalar(raw: str | None) -> float | str | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return raw


def _load_summary(path: Path) -> Dict[str, Dict[str, float | str | None]]:
    rows = _load_csv(path)
    out: Dict[str, Dict[str, float | str | None]] = {}
    for row in rows:
        model = row["model"]
        parsed: Dict[str, float | str | None] = {}
        for key, value in row.items():
            if key in {"model", "checkpoint"}:
                parsed[key] = value  # type: ignore[assignment]
            else:
                parsed[key] = _parse_scalar(value)
        out[model] = parsed
    return out


def _load_per_seed(path: Path) -> Dict[str, List[Dict[str, float | int | str | None]]]:
    rows = _load_csv(path)
    out: Dict[str, List[Dict[str, float | int | str | None]]] = {}
    for row in rows:
        model = row["model"]
        parsed: Dict[str, float | int | str | None] = {"model": model}
        for key, value in row.items():
            if key == "model":
                continue
            if key in {"checkpoint"}:
                parsed[key] = value
            elif key in {"seed", "episodes"}:
                parsed[key] = int(value)
            else:
                parsed[key] = _parse_scalar(value)
        out.setdefault(model, []).append(parsed)
    return out


def _metric_summary(metric: Dict[str, str], summary: Dict[str, Dict[str, float | None]]) -> Dict[str, object]:
    key = metric["key"]
    rows = []
    valid_values = []
    for model, model_row in summary.items():
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        mean_value = model_row.get(mean_key)
        std_value = model_row.get(std_key)
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "color": MODEL_COLORS.get(model, "#64748b"),
                "mean": mean_value,
                "std": std_value,
            }
        )
        if isinstance(mean_value, float):
            valid_values.append((model, mean_value))
    best_model = None
    if valid_values:
        choose = max if metric["direction"] == "higher" else min
        best_model = choose(valid_values, key=lambda item: item[1])[0]
    return {
        **metric,
        "rows": rows,
        "best_model": best_model,
    }


def _takeaway(label: str, metric_key: str, summary: Dict[str, Dict[str, float | None]], direction: str) -> Dict[str, object]:
    candidates = []
    for model, row in summary.items():
        value = row.get(f"{metric_key}_mean")
        if isinstance(value, float):
            candidates.append((model, value))
    if not candidates:
        return {"label": label, "model": None, "value": None}
    choose = max if direction == "higher" else min
    model, value = choose(candidates, key=lambda item: item[1])
    return {
        "label": label,
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "value": value,
        "color": MODEL_COLORS.get(model, "#64748b"),
    }


def _table_rows(summary: Dict[str, Dict[str, float | None]], metrics: List[Dict[str, str]]) -> List[Dict[str, object]]:
    rows = []
    for model, data in summary.items():
        row = {
            "model": model,
            "label": MODEL_LABELS.get(model, model),
        }
        for metric in metrics:
            key = metric["key"]
            row[key] = {
                "mean": data.get(f"{key}_mean"),
                "std": data.get(f"{key}_std"),
            }
        rows.append(row)
    return rows


def build_report(
    summary_csv: Path,
    per_seed_csv: Path,
    output_html: Path,
    *,
    compare_return: bool = False,
    title: str = "Three-Model Behavior Comparison",
    subtitle: str = "risk_base vs safe vs off",
) -> Path:
    summary = _load_summary(summary_csv)
    per_seed = _load_per_seed(per_seed_csv)

    metrics = ([] if not compare_return else RETURN_METRICS) + TASK_METRICS + SAFETY_METRICS + SHIELD_METRICS
    notes = (
        [
            "This comparison includes episode_return because all compared checkpoints were trained under aligned DPM-reward normalization.",
            "Return is now comparable across the displayed models, but task and safety metrics should still be interpreted as the primary behavior view.",
        ]
        if compare_return
        else [
            "This comparison intentionally does not use episode_return as a primary ranking metric.",
            "Reason: the older baselines and the newer risk_base run were trained under different DPM-reward normalization settings, so return scale is not directly comparable.",
            "Primary interpretation should use task, safety, and shield-behavior metrics.",
        ]
    )
    report = {
        "title": title,
        "subtitle": subtitle,
        "summary_csv": str(summary_csv.resolve()),
        "per_seed_csv": str(per_seed_csv.resolve()),
        "compare_return": bool(compare_return),
        "models": [
            {
                "name": model,
                "label": MODEL_LABELS.get(model, model),
                "color": MODEL_COLORS.get(model, "#64748b"),
                "checkpoint": summary[model].get("checkpoint"),
                "num_seeds": summary[model].get("num_seeds"),
                "episodes_per_seed": summary[model].get("episodes_per_seed"),
                "per_seed_count": len(per_seed.get(model, [])),
            }
            for model in summary.keys()
        ],
        "notes": notes,
        "takeaways": [
            _takeaway("Best Search Rate", "search_rate", summary, "higher"),
            _takeaway("Lowest Collision Count", "collision_count", summary, "lower"),
            _takeaway("Lowest Near-Miss Rate", "near_miss_rate", summary, "lower"),
        ],
        "return_metrics": ([] if not compare_return else [_metric_summary(metric, summary) for metric in RETURN_METRICS]),
        "task_metrics": [_metric_summary(metric, summary) for metric in TASK_METRICS],
        "safety_metrics": [_metric_summary(metric, summary) for metric in SAFETY_METRICS],
        "shield_metrics": [_metric_summary(metric, summary) for metric in SHIELD_METRICS],
        "table_rows": _table_rows(summary, metrics),
    }

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(_render_html(report), encoding="utf-8")

    json_path = output_html.with_suffix(".json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_html


def _render_html(report: Dict[str, object]) -> str:
    data = json.dumps(report, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{report['title']}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: rgba(255,255,255,0.76);
      --panel-strong: rgba(255,255,255,0.92);
      --ink: #1f2937;
      --muted: #6b7280;
      --line: rgba(148, 163, 184, 0.28);
      --accent: #1d4ed8;
      --safe: #0f766e;
      --off: #b45309;
      --shadow: 0 18px 38px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(59,130,246,0.08), transparent 32%),
        radial-gradient(circle at top right, rgba(180,83,9,0.08), transparent 30%),
        linear-gradient(180deg, #f8f4ec 0%, #f2ecdf 100%);
      color: var(--ink);
    }}
    .page {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px 24px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92));
      border: 1px solid rgba(148,163,184,0.22);
      border-radius: 28px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
      margin-bottom: 22px;
    }}
    .eyebrow {{
      display: inline-block;
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #1d4ed8;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.15;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 16px;
      max-width: 900px;
      line-height: 1.6;
    }}
    .paths {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .path-card {{
      background: rgba(248,250,252,0.88);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
      font-size: 13px;
    }}
    .path-card strong {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      word-break: break-all;
      font-size: 12px;
    }}
    .notes {{
      margin: 16px 0 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.6;
    }}
    .section {{
      margin-top: 22px;
    }}
    .section-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .section-header h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .section-header p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .takeaway-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    .takeaway-card, .metric-card, .table-card {{
      background: var(--panel);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255,255,255,0.56);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .takeaway-card {{
      padding: 18px 18px 16px;
    }}
    .takeaway-label {{
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .takeaway-model {{
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .takeaway-value {{
      font-size: 14px;
      color: var(--muted);
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .metric-card {{
      padding: 18px;
    }}
    .metric-card h3 {{
      margin: 0 0 6px;
      font-size: 18px;
    }}
    .metric-meta {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 14px;
      line-height: 1.55;
    }}
    .chart {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 116px 1fr 88px;
      gap: 10px;
      align-items: center;
    }}
    .bar-label {{
      font-size: 13px;
      font-weight: 600;
    }}
    .bar-track {{
      position: relative;
      height: 16px;
      background: rgba(226,232,240,0.72);
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      border-radius: 999px;
    }}
    .bar-best {{
      outline: 2px solid rgba(15,23,42,0.22);
      outline-offset: 2px;
    }}
    .bar-value {{
      text-align: right;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      color: var(--muted);
    }}
    .legend {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: middle;
      background: var(--swatch, #94a3b8);
    }}
    .table-card {{
      padding: 18px;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 940px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 13px;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .model-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
    }}
    .model-chip::before {{
      content: "";
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--chip-color, #64748b);
      display: inline-block;
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 760px) {{
      .page {{ padding: 18px 14px 36px; }}
      .hero {{ padding: 22px 18px; border-radius: 22px; }}
      h1 {{ font-size: 28px; }}
      .bar-row {{ grid-template-columns: 92px 1fr 74px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Behavior Eval</div>
      <h1>{report['title']}</h1>
      <p class="subtitle">固定已有 checkpoint，使用统一多 seed evaluation 对比模型的任务表现、安全代价与 shield 行为差异。</p>
      <div class="paths">
        <div class="path-card"><strong>Summary CSV</strong><code>{report['summary_csv']}</code></div>
        <div class="path-card"><strong>Per-Seed CSV</strong><code>{report['per_seed_csv']}</code></div>
        <div class="path-card"><strong>页面数据 JSON</strong><code>{str(Path(report['summary_csv']).with_name(Path(report['summary_csv']).stem.replace('summary_metrics', 'model_compare_report')).resolve()) if False else 'Embedded in page + sibling JSON'}</code></div>
      </div>
      <ul class="notes" id="notes"></ul>
    </section>

    <section class="section">
      <div class="section-header">
        <h2>Quick Takeaways</h2>
        <p>先看最直观的赢家，不用先翻表。</p>
      </div>
      <div class="takeaway-grid" id="takeaways"></div>
    </section>

    <section class="section" id="return-section" style="display:none;">
      <div class="section-header">
        <h2>Return Metrics</h2>
        <p>本页 reward normalization 已对齐，因此可以直接比较 return。</p>
      </div>
      <div class="metric-grid" id="return-metrics"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <h2>Task Metrics</h2>
        <p>更偏任务完成质量。方向已在卡片内标注。</p>
      </div>
      <div class="metric-grid" id="task-metrics"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <h2>Safety Metrics</h2>
        <p>更偏碰撞与危险接近，主结论优先看这一组。</p>
      </div>
      <div class="metric-grid" id="safety-metrics"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <h2>Shield Behavior</h2>
        <p>看 intervention 频率，以及允许动作集的收缩情况。</p>
      </div>
      <div class="metric-grid" id="shield-metrics"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <h2>Summary Table</h2>
        <p>均值 ± 标准差，方便精读与截图。</p>
      </div>
      <div class="table-card">
        <table id="summary-table"></table>
      </div>
    </section>
  </div>

  <script>
    const data = {data};

    function fmtValue(value, unit) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
      if (unit === "ratio") return (value * 100).toFixed(1) + "%";
      if (Math.abs(value) >= 100) return value.toFixed(1);
      if (Math.abs(value) >= 10) return value.toFixed(2);
      return value.toFixed(3);
    }}

    function fmtCell(mean, std, unit) {{
      if (mean === null || mean === undefined || Number.isNaN(mean)) return "N/A";
      const meanText = fmtValue(mean, unit);
      if (std === null || std === undefined || Number.isNaN(std)) return meanText;
      const stdText = unit === "ratio" ? (std * 100).toFixed(1) + "%" : std.toFixed(2);
      return `${{meanText}} ± ${{stdText}}`;
    }}

    function renderNotes() {{
      const el = document.getElementById("notes");
      el.innerHTML = data.notes.map(note => `<li>${{note}}</li>`).join("");
    }}

    function renderTakeaways() {{
      const el = document.getElementById("takeaways");
      el.innerHTML = data.takeaways.map(item => {{
        if (!item.model) {{
          return `<div class="takeaway-card"><div class="takeaway-label">${{item.label}}</div><div class="takeaway-model">N/A</div></div>`;
        }}
        return `
          <div class="takeaway-card">
            <div class="takeaway-label">${{item.label}}</div>
            <div class="takeaway-model" style="color:${{item.color}}">${{item.model_label}}</div>
            <div class="takeaway-value">${{fmtValue(item.value, item.label.includes("Count") ? "count / episode" : "ratio")}}</div>
          </div>
        `;
      }}).join("");
    }}

    function renderMetricSection(containerId, metrics) {{
      const root = document.getElementById(containerId);
      root.innerHTML = metrics.map(metric => {{
        const values = metric.rows.filter(row => row.mean !== null).map(row => row.mean);
        const maxValue = values.length ? Math.max(...values) : 1;
        const bars = metric.rows.map(row => {{
          const width = row.mean === null ? 0 : (maxValue <= 0 ? 0 : (row.mean / maxValue) * 100);
          const bestClass = row.model === metric.best_model ? "bar-best" : "";
          return `
            <div class="bar-row">
              <div class="bar-label">${{row.label}}</div>
              <div class="bar-track ${{bestClass}}">
                <div class="bar-fill" style="width:${{width}}%; background:${{row.color}};"></div>
              </div>
              <div class="bar-value">${{fmtCell(row.mean, row.std, metric.unit)}}</div>
            </div>
          `;
        }}).join("");
        return `
          <article class="metric-card">
            <h3>${{metric.label}}</h3>
            <div class="metric-meta">${{metric.description}}<br />方向：<strong>${{metric.direction === "higher" ? "越高越好" : "越低越好"}}</strong>，单位：<strong>${{metric.unit}}</strong></div>
            <div class="chart">${{bars}}</div>
          </article>
        `;
      }}).join("");
    }}

    function renderTable() {{
      const table = document.getElementById("summary-table");
      const columns = [
        ...data.task_metrics,
        ...data.safety_metrics,
        ...data.shield_metrics,
      ];
      const head = `
        <thead>
          <tr>
            <th>Model</th>
            ${{columns.map(col => `<th>${{col.label}}</th>`).join("")}}
          </tr>
        </thead>
      `;
      const body = data.table_rows.map(row => {{
        const color = data.models.find(model => model.name === row.model)?.color || "#64748b";
        const cells = columns.map(col => {{
          const cell = row[col.key];
          return `<td>${{fmtCell(cell.mean, cell.std, col.unit)}}</td>`;
        }}).join("");
        return `
          <tr>
            <td><span class="model-chip" style="--chip-color:${{color}}">${{row.label}}</span></td>
            ${{cells}}
          </tr>
        `;
      }}).join("");
      table.innerHTML = head + `<tbody>${{body}}</tbody>`;
    }}

    renderNotes();
    renderTakeaways();
    if (data.compare_return) {{
      document.getElementById("return-section").style.display = "";
      renderMetricSection("return-metrics", data.return_metrics);
    }}
    renderMetricSection("task-metrics", data.task_metrics);
    renderMetricSection("safety-metrics", data.safety_metrics);
    renderMetricSection("shield-metrics", data.shield_metrics);
    renderTable();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a three-model comparison HTML page from multiseed eval CSVs.")
    parser.add_argument("--summary-csv", required=True, type=str)
    parser.add_argument("--per-seed-csv", required=True, type=str)
    parser.add_argument("--output-html", required=True, type=str)
    parser.add_argument("--compare-return", action="store_true", help="Include episode_return/avg_reward metrics and aligned-normalization note.")
    parser.add_argument("--title", type=str, default="Three-Model Behavior Comparison")
    parser.add_argument("--subtitle", type=str, default="risk_base vs safe vs off")
    args = parser.parse_args()

    output = build_report(
        summary_csv=Path(args.summary_csv),
        per_seed_csv=Path(args.per_seed_csv),
        output_html=Path(args.output_html),
        compare_return=bool(args.compare_return),
        title=args.title,
        subtitle=args.subtitle,
    )
    print(f"[model-compare-page] html={output}")
    print(f"[model-compare-page] json={output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
