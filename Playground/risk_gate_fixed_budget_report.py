from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DISPLAY_ORDER = [
    "baseline_v1",
    "ablation_hist0",
    "v2_proposed_action_clearance",
    "v3_hybrid_clear",
    "v3_hybrid_clear_fragility",
    "v_next_prop_gap_support_region",
    "v_next2_prop_fragility_support_region",
    "cand_clearmin_prop_fragility_support_region",
    "cand_clearmin_prop_support_region",
    "vnext_tune_proxy_prop_gap_region",
    "vnext_tune_exact_prop_support",
    "vnext_tune_exact_prop_only",
]

DISPLAY_META = {
    "baseline_v1": {
        "label": "baseline_v1",
        "short": "baseline v1",
        "formula": "clear_min + region + hist",
    },
    "ablation_hist0": {
        "label": "ablation_hist0",
        "short": "hist=0",
        "formula": "clear_min + region",
    },
    "v2_proposed_action_clearance": {
        "label": "v2_proposed_action_clearance",
        "short": "v2 prop clear",
        "formula": "prop_clear + region + hist",
    },
    "v3_hybrid_clear": {
        "label": "v3_hybrid_clear",
        "short": "v3 hybrid clear",
        "formula": "clear_min + clear_prop + region",
    },
    "v3_hybrid_clear_fragility": {
        "label": "v3_hybrid_clear_fragility",
        "short": "v3 + fragility",
        "formula": "clear_min + clear_prop + fragility + region",
    },
    "v_next_prop_gap_support_region": {
        "label": "v_next_prop_gap_support_region",
        "short": "v_next",
        "formula": "prop_clear + clear_gap + support + region",
    },
    "v_next2_prop_fragility_support_region": {
        "label": "v_next2_prop_fragility_support_region",
        "short": "v_next2",
        "formula": "prop_clear + fragility + support + region",
    },
    "cand_clearmin_prop_fragility_support_region": {
        "label": "cand_clearmin_prop_fragility_support_region",
        "short": "cmin+cprop+frag+sup",
        "formula": "clear_min + clear_prop + fragility + support + region",
    },
    "cand_clearmin_prop_support_region": {
        "label": "cand_clearmin_prop_support_region",
        "short": "cmin+cprop+sup",
        "formula": "clear_min + clear_prop + support + region",
    },
    "vnext_tune_proxy_prop_gap_region": {
        "label": "vnext_tune_proxy_prop_gap_region",
        "short": "vnext proxy",
        "formula": "prop_clear + clear_gap + region",
    },
    "vnext_tune_exact_prop_support": {
        "label": "vnext_tune_exact_prop_support",
        "short": "vnext exact p+s",
        "formula": "prop_clear + support",
    },
    "vnext_tune_exact_prop_only": {
        "label": "vnext_tune_exact_prop_only",
        "short": "vnext exact prop",
        "formula": "prop_clear",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed eligible-gate-budget comparison report from offline risk-validation records."
    )
    parser.add_argument("--records-csv", required=True, help="Offline fixed-safe records CSV with variant rows.")
    parser.add_argument("--source-json", type=str, default=None, help="Optional offline validation JSON for metadata.")
    parser.add_argument(
        "--budget-rates",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6],
        help="Eligible-only gate budgets to compare.",
    )
    parser.add_argument(
        "--include-variants",
        type=str,
        nargs="*",
        default=None,
        help="Optional variant-name allowlist. When set, only these variants are shown in the report.",
    )
    parser.add_argument("--page-title", type=str, default="20% / 40% / 60% eligible gate budget 对比")
    parser.add_argument("--page-badge", type=str, default="Fixed Eligible-Gate Budget")
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    parser.add_argument("--output-html", required=True, help="Output HTML path.")
    return parser.parse_args()


def load_records(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row_idx, row in enumerate(csv.DictReader(f)):
            payload = dict(row)
            payload["row_id"] = int(row_idx)
            payload["variant_score"] = float(row["variant_score"])
            payload["eligible"] = int(row["eligible"])
            payload["need_rec"] = int(row["need_rec"])
            payload["proposed_in_a_hard"] = int(row.get("proposed_in_a_hard", payload["eligible"]))
            rows.append(payload)
    return rows


def exact_budget_metrics(rows: List[Dict[str, Any]], budget_rate: float) -> Dict[str, Any]:
    eligible_rows = [row for row in rows if int(row["eligible"]) == 1]
    eligible_rows.sort(key=lambda row: (-float(row["variant_score"]), int(row["row_id"])))
    eligible_count = len(eligible_rows)
    need_rec_count = sum(int(row["need_rec"]) for row in eligible_rows)
    if eligible_count == 0:
        return {
            "budget_target": float(budget_rate),
            "selected_eligible_count": 0,
            "actual_eligible_gate_rate": 0.0,
            "eligible_precision_need_rec": 0.0,
            "recall_need_rec": 0.0,
            "score_cutoff": None,
        }

    selected_count = min(eligible_count, max(1, int(round(float(budget_rate) * eligible_count))))
    selected_rows = eligible_rows[:selected_count]
    tp = sum(int(row["need_rec"]) for row in selected_rows)
    return {
        "budget_target": float(budget_rate),
        "selected_eligible_count": int(selected_count),
        "actual_eligible_gate_rate": float(selected_count / eligible_count),
        "eligible_precision_need_rec": float(tp / selected_count) if selected_count else 0.0,
        "recall_need_rec": float(tp / need_rec_count) if need_rec_count else 0.0,
        "score_cutoff": float(selected_rows[-1]["variant_score"]) if selected_rows else None,
    }


def threshold_proxy_metrics(rows: List[Dict[str, Any]], budget_rate: float) -> Dict[str, Any]:
    if not rows:
        return {
            "budget_target": float(budget_rate),
            "threshold": None,
            "gate_rate": 0.0,
            "eligible_gate_rate": 0.0,
            "eligible_precision_need_rec": 0.0,
            "recall_need_rec": 0.0,
            "wasted_gate_rate": 0.0,
        }

    total = len(rows)
    eligible_count = sum(int(row["eligible"]) for row in rows)
    need_rec_count = sum(int(row["need_rec"]) for row in rows)
    unique_scores = sorted({float(row["variant_score"]) for row in rows}, reverse=True)
    candidates = [unique_scores[0] + 1e-9] + unique_scores + [unique_scores[-1] - 1e-9]

    best_choice: Dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for threshold in candidates:
        predicted = [row for row in rows if float(row["variant_score"]) >= float(threshold)]
        predicted_eligible = [row for row in predicted if int(row["eligible"]) == 1]
        predicted_eligible_count = len(predicted_eligible)
        tp = sum(int(row["need_rec"]) for row in predicted)
        eligible_gate_rate = float(predicted_eligible_count / max(eligible_count, 1))
        wasted_gate_rate = float((len(predicted) - predicted_eligible_count) / max(total, 1))
        precision = float(tp / predicted_eligible_count) if predicted_eligible_count else 0.0
        recall = float(tp / need_rec_count) if need_rec_count else 0.0
        ranking_key = (
            abs(eligible_gate_rate - float(budget_rate)),
            wasted_gate_rate,
            -recall,
            -precision,
            -float(threshold),
        )
        if best_key is None or ranking_key < best_key:
            best_key = ranking_key
            best_choice = {
                "budget_target": float(budget_rate),
                "threshold": float(threshold),
                "gate_rate": float(len(predicted) / max(total, 1)),
                "eligible_gate_rate": eligible_gate_rate,
                "eligible_precision_need_rec": precision,
                "recall_need_rec": recall,
                "wasted_gate_rate": wasted_gate_rate,
            }

    assert best_choice is not None
    return best_choice


def build_report(
    records: List[Dict[str, Any]],
    source_meta: Dict[str, Any],
    budget_rates: List[float],
    *,
    include_variants: List[str] | None = None,
    page_title: str,
    page_badge: str,
) -> Dict[str, Any]:
    allowed = {str(name) for name in include_variants} if include_variants else None
    variant_names = list({str(row["variant"]) for row in records})
    if allowed is not None:
        variant_names = [name for name in variant_names if name in allowed]
    ordered_variant_names = [name for name in DISPLAY_ORDER if name in variant_names]
    ordered_variant_names.extend(sorted(name for name in variant_names if name not in ordered_variant_names))

    variant_reports: List[Dict[str, Any]] = []
    for name in ordered_variant_names:
        subset = [row for row in records if str(row["variant"]) == name]
        meta = DISPLAY_META.get(
            name,
            {"label": name, "short": name, "formula": "unknown"},
        )
        exact = [exact_budget_metrics(subset, budget) for budget in budget_rates]
        proxy = [threshold_proxy_metrics(subset, budget) for budget in budget_rates]
        variant_reports.append(
            {
                "name": name,
                "label": meta["label"],
                "short": meta["short"],
                "formula": meta["formula"],
                "agent_step_count": int(len(subset)),
                "eligible_agent_step_count": int(sum(int(row["eligible"]) for row in subset)),
                "need_rec_count": int(sum(int(row["need_rec"]) for row in subset)),
                "exact_budget_metrics": exact,
                "threshold_proxy_metrics": proxy,
            }
        )

    budget_summaries: List[Dict[str, Any]] = []
    for budget in budget_rates:
        exact_rows = [
            {
                "name": variant["name"],
                "short": variant["short"],
                **next(item for item in variant["exact_budget_metrics"] if float(item["budget_target"]) == float(budget)),
            }
            for variant in variant_reports
        ]
        best_precision_value = max(float(row["eligible_precision_need_rec"]) for row in exact_rows)
        best_recall_value = max(float(row["recall_need_rec"]) for row in exact_rows)
        precision_ties = [
            str(row["short"]) for row in exact_rows if abs(float(row["eligible_precision_need_rec"]) - best_precision_value) <= 1e-12
        ]
        recall_ties = [
            str(row["short"]) for row in exact_rows if abs(float(row["recall_need_rec"]) - best_recall_value) <= 1e-12
        ]
        collapsed = (
            max(float(row["eligible_precision_need_rec"]) for row in exact_rows)
            - min(float(row["eligible_precision_need_rec"]) for row in exact_rows)
            <= 1e-12
            and max(float(row["recall_need_rec"]) for row in exact_rows)
            - min(float(row["recall_need_rec"]) for row in exact_rows)
            <= 1e-12
        )
        budget_summaries.append(
            {
                "budget_target": float(budget),
                "best_precision_variants": precision_ties,
                "best_precision_value": float(best_precision_value),
                "best_recall_variants": recall_ties,
                "best_recall_value": float(best_recall_value),
                "budget_collapsed": bool(collapsed),
            }
        )

    trajectory_summary = source_meta.get("trajectory_summary", {})
    return {
        "checkpoint": source_meta.get("checkpoint", ""),
        "episodes": int(source_meta.get("episodes", 0)),
        "env_steps": int(source_meta.get("env_steps", 0)),
        "source_json": source_meta.get("_source_json", ""),
        "source_csv": source_meta.get("_source_csv", ""),
        "page_title": str(page_title),
        "page_badge": str(page_badge),
        "budget_rates": [float(budget) for budget in budget_rates],
        "trajectory_summary": trajectory_summary,
        "variant_reports": variant_reports,
        "budget_summaries": budget_summaries,
        "notes": [
            "Primary view is exact eligible-only top-k budget: for each variant we sort eligible agent-steps by risk score and take the top budget fraction.",
            "This exact-budget view is a ranking diagnostic, not a single-threshold runtime policy, so it neutralizes the 'high threshold -> tiny gate -> inflated precision' effect.",
            "The threshold-proxy rows are provided separately to show how closely a single global threshold could realize the same eligible budget in runtime.",
            "A_hard-empty and ineligible-but-nonempty samples remain outside the exact-budget ranking pool by design, consistent with the eligible-only validation semantics.",
        ],
    }


def render_html(payload: Dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Risk Fixed Budget Comparison</title>
  <style>
    :root {{
      --bg: #f6f1e7;
      --panel: #fffaf2;
      --ink: #22313a;
      --muted: #5d6972;
      --line: #d9ccb9;
      --accent: #b14d2f;
      --good: #2f7d6d;
      --cool: #295f8a;
      --warn: #96583b;
      --shadow: 0 12px 28px rgba(46, 34, 18, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(177, 77, 47, 0.10), transparent 26%),
        radial-gradient(circle at top right, rgba(41, 95, 138, 0.10), transparent 24%),
        linear-gradient(180deg, #fbf6ee 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 28px 20px 48px; }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 28px; }}
    .panel {{ padding: 20px; margin-top: 18px; }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{ font-size: 34px; }}
    h2 {{ font-size: 20px; margin-bottom: 12px; }}
    h3 {{ font-size: 16px; margin-bottom: 10px; }}
    p, li {{ color: var(--muted); line-height: 1.65; }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      background: rgba(35,49,58,0.06);
      padding: 2px 6px;
      border-radius: 6px;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(177, 77, 47, 0.1);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .box {{
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(35,49,58,0.08);
      border-radius: 16px;
      padding: 14px 16px;
    }}
    .box strong {{ display: block; margin-bottom: 8px; }}
    .box .value {{ font-size: 28px; font-weight: 800; margin-top: 6px; }}
    .box .sub {{ font-size: 13px; margin-top: 8px; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 18px;
      margin-top: 18px;
    }}
    .span-12 {{ grid-column: span 12; }}
    .span-6 {{ grid-column: span 6; }}
    @media (max-width: 960px) {{
      .span-6 {{ grid-column: span 12; }}
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid rgba(90,103,112,0.12);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    tr.best td {{ background: rgba(47,125,109,0.09); }}
    tr.warn td {{ background: rgba(177,77,47,0.07); }}
    .chart {{
      height: 300px;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(246,238,226,0.95));
      border: 1px solid rgba(82, 96, 109, 0.12);
      padding: 12px;
    }}
    .chart svg {{
      width: 100%;
      height: 100%;
      overflow: visible;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .dot {{
      width: 11px;
      height: 11px;
      border-radius: 999px;
    }}
    ul {{ margin: 10px 0 0; padding-left: 18px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <span class="badge">{payload["page_badge"]}</span>
      <h1 style="margin-top: 12px;">{payload["page_title"]}</h1>
      <p style="margin-top: 10px;">
        这页把主验证口径固定成 <code>eligible-only</code> 的 exact top-k budget：
        每个 variant 只在 <code>proposed_action ∈ A_hard</code> 的样本里，按 risk score 排序后取前
        <code>20%</code> / <code>40%</code> / <code>60%</code>。这样可以直接看排序能力，
        不再被“阈值过高导致 gate 太少，从而虚高 precision”的现象误导。
      </p>
      <div class="meta" id="hero-meta"></div>
    </section>

    <div class="grid">
      <section class="panel span-6">
        <h2>Exact Budget Precision</h2>
        <p>横轴：eligible gate budget，单位为 %。纵轴：eligible precision，单位为 %。</p>
        <div class="chart" id="precision-chart"></div>
        <div class="legend" id="legend-a"></div>
      </section>

      <section class="panel span-6">
        <h2>Exact Budget Recall</h2>
        <p>横轴：eligible gate budget，单位为 %。纵轴：recall，单位为 %。</p>
        <div class="chart" id="recall-chart"></div>
        <div class="legend" id="legend-b"></div>
      </section>

      <section class="panel span-12">
        <h2>Exact Top-k Budget Tables</h2>
        <div id="budget-tables"></div>
      </section>

      <section class="panel span-12">
        <h2>Closest Threshold Runtime Proxy</h2>
        <p>
          下面这组表不是 exact top-k 排序，而是问：如果仍坚持用单一阈值 <code>eta</code>，
          那么哪个 threshold 最接近目标 eligible budget，以及它会带来多少 wasted gate。
        </p>
        <div id="proxy-tables"></div>
      </section>

      <section class="panel span-12">
        <h2>主要结论</h2>
        <ul id="takeaways"></ul>
      </section>
    </div>
  </div>

  <script>
    const data = {data_json};
    const COLORS = {{
      baseline_v1: "#b14d2f",
      ablation_hist0: "#2f7d6d",
      v3_hybrid_clear: "#8c6a1f",
      v3_hybrid_clear_fragility: "#295f8a",
      v_next_prop_gap_support_region: "#7a3348",
      v_next2_prop_fragility_support_region: "#5a3fc0",
      cand_clearmin_prop_fragility_support_region: "#1f7a8c",
      cand_clearmin_prop_support_region: "#8f6f2a",
      vnext_tune_proxy_prop_gap_region: "#4b5d16",
      vnext_tune_exact_prop_support: "#0f766e",
      vnext_tune_exact_prop_only: "#b54708",
    }};

    const pct = value => `${{(value * 100).toFixed(1)}}%`;
    const fmt = value => value == null ? "-" : Number(value).toFixed(3);

    const traj = data.trajectory_summary || {{}};
    document.getElementById("hero-meta").innerHTML = [
      `<div class="box"><strong>Checkpoint</strong>${{data.checkpoint}}</div>`,
      `<div class="box"><strong>Trajectory</strong>${{data.episodes}} episodes, ${{data.env_steps}} env steps</div>`,
      `<div class="box"><strong>Eligible Rate</strong>${{pct(traj.eligible_agent_step_rate || 0)}}</div>`,
      `<div class="box"><strong>Need Rec Rate</strong>${{pct(traj.need_rec_rate || 0)}}</div>`,
      `<div class="box"><strong>dead_end_hard</strong>${{pct(traj.dead_end_hard_agent_step_rate || 0)}}</div>`,
      `<div class="box"><strong>Ineligible Nonempty</strong>${{pct(traj.ineligible_nonempty_agent_step_rate || 0)}}</div>`,
    ].join("");

    function renderLegend(elId, items) {{
      document.getElementById(elId).innerHTML = items.map(item => `
        <span class="legend-item"><span class="dot" style="background:${{item.color}};"></span>${{item.label}}</span>
      `).join("");
    }}

    function renderLineChart(el, series, labels) {{
      const width = 680;
      const height = 260;
      const pad = {{ top: 18, right: 18, bottom: 34, left: 42 }};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const allValues = series.flatMap(s => s.values);
      const maxVal = Math.max(...allValues, 1e-6);
      const x = i => pad.left + (plotW * i) / Math.max(labels.length - 1, 1);
      const y = v => pad.top + plotH - (v / maxVal) * plotH;

      const axes = `
        <line x1="${{pad.left}}" y1="${{pad.top}}" x2="${{pad.left}}" y2="${{pad.top + plotH}}" stroke="#9fb3c8" stroke-width="1.2" />
        <line x1="${{pad.left}}" y1="${{pad.top + plotH}}" x2="${{pad.left + plotW}}" y2="${{pad.top + plotH}}" stroke="#9fb3c8" stroke-width="1.2" />
      `;
      const ticks = [0, maxVal / 2, maxVal].map(v => `
        <g>
          <line x1="${{pad.left}}" y1="${{y(v)}}" x2="${{pad.left + plotW}}" y2="${{y(v)}}" stroke="rgba(159,179,200,0.35)" stroke-dasharray="4 4" />
          <text x="${{pad.left - 8}}" y="${{y(v) + 4}}" text-anchor="end" font-size="11" fill="#52606d">${{pct(v)}}</text>
        </g>
      `).join("");
      const xLabels = labels.map((label, i) => `
        <text x="${{x(i)}}" y="${{pad.top + plotH + 22}}" text-anchor="middle" font-size="11" fill="#52606d">${{label}}</text>
      `).join("");
      const paths = series.map(s => {{
        const d = s.values.map((v, i) => `${{i === 0 ? "M" : "L"}} ${{x(i)}} ${{y(v)}}`).join(" ");
        const points = s.values.map((v, i) => `<circle cx="${{x(i)}}" cy="${{y(v)}}" r="4" fill="${{s.color}}" />`).join("");
        return `<path d="${{d}}" fill="none" stroke="${{s.color}}" stroke-width="3" stroke-linecap="round" />${{points}}`;
      }}).join("");
      el.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img">${{axes}}${{ticks}}${{xLabels}}${{paths}}</svg>`;
    }}

    const labels = data.budget_rates.map(v => pct(v));
    const precisionSeries = data.variant_reports.map(item => ({{
      label: item.short,
      color: COLORS[item.name] || "#52606d",
      values: item.exact_budget_metrics.map(row => row.eligible_precision_need_rec),
    }}));
    const recallSeries = data.variant_reports.map(item => ({{
      label: item.short,
      color: COLORS[item.name] || "#52606d",
      values: item.exact_budget_metrics.map(row => row.recall_need_rec),
    }}));
    renderLineChart(document.getElementById("precision-chart"), precisionSeries, labels);
    renderLineChart(document.getElementById("recall-chart"), recallSeries, labels);
    renderLegend("legend-a", precisionSeries);
    renderLegend("legend-b", recallSeries);

    document.getElementById("budget-tables").innerHTML = data.budget_rates.map(budget => {{
      const rows = data.variant_reports.map(variant => {{
        const row = variant.exact_budget_metrics.find(item => item.budget_target === budget);
        return {{ ...variant, ...row }};
      }});
      const bestRecall = Math.max(...rows.map(row => row.recall_need_rec));
      const bestPrecision = Math.max(...rows.map(row => row.eligible_precision_need_rec));
      return `
        <h3>Budget = ${{pct(budget)}}</h3>
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              <th>Formula</th>
              <th>Eligible Budget</th>
              <th>Selected Eligible</th>
              <th>Eligible Precision</th>
              <th>Recall</th>
              <th>Score Cutoff</th>
            </tr>
          </thead>
          <tbody>
            ${{rows.map(row => `
              <tr class="${{row.recall_need_rec === bestRecall || row.eligible_precision_need_rec === bestPrecision ? "best" : ""}}">
                <td><strong>${{row.label}}</strong></td>
                <td>${{row.formula}}</td>
                <td>${{pct(row.actual_eligible_gate_rate)}}</td>
                <td>${{row.selected_eligible_count}}</td>
                <td>${{pct(row.eligible_precision_need_rec)}}</td>
                <td>${{pct(row.recall_need_rec)}}</td>
                <td>${{fmt(row.score_cutoff)}}</td>
              </tr>
            `).join("")}}
          </tbody>
        </table>
      `;
    }}).join("");

    document.getElementById("proxy-tables").innerHTML = data.budget_rates.map(budget => {{
      const rows = data.variant_reports.map(variant => {{
        const row = variant.threshold_proxy_metrics.find(item => item.budget_target === budget);
        return {{ ...variant, ...row }};
      }});
      return `
        <h3>Budget = ${{pct(budget)}}</h3>
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              <th>Threshold</th>
              <th>Actual Eligible Gate</th>
              <th>Gate Rate</th>
              <th>Eligible Precision</th>
              <th>Recall</th>
              <th>Wasted Gate</th>
            </tr>
          </thead>
          <tbody>
            ${{rows.map(row => `
              <tr>
                <td><strong>${{row.label}}</strong></td>
                <td>${{fmt(row.threshold)}}</td>
                <td>${{pct(row.eligible_gate_rate)}}</td>
                <td>${{pct(row.gate_rate)}}</td>
                <td>${{pct(row.eligible_precision_need_rec)}}</td>
                <td>${{pct(row.recall_need_rec)}}</td>
                <td>${{pct(row.wasted_gate_rate)}}</td>
              </tr>
            `).join("")}}
          </tbody>
        </table>
      `;
    }}).join("");

    const focus20 = data.budget_summaries.find(item => item.budget_target === 0.2);
    const focus40 = data.budget_summaries.find(item => item.budget_target === 0.4);
    const focus60 = data.budget_summaries.find(item => item.budget_target === 0.6);
    const joinVariants = list => list.join(" / ");
    document.getElementById("takeaways").innerHTML = [
      `在 20% exact eligible budget 下，recall 最好的是 ${{joinVariants(focus20.best_recall_variants)}}，precision 最好的是 ${{joinVariants(focus20.best_precision_variants)}}。`,
      `40% budget: ${{focus40.budget_collapsed ? "几乎所有 variant 完全并列，说明排序在更深预算区间已经塌缩。" : `best recall = ${{joinVariants(focus40.best_recall_variants)}}，best precision = ${{joinVariants(focus40.best_precision_variants)}}。`}}`,
      `60% budget: ${{focus60.budget_collapsed ? "几乎所有 variant 完全并列，说明 score 大面积进入平原区，无法继续区分 need_rec。" : `best recall = ${{joinVariants(focus60.best_recall_variants)}}，best precision = ${{joinVariants(focus60.best_precision_variants)}}。`}}`,
      `threshold proxy 表可以用来检查：某个 variant 即便在 exact budget 下表现不错，也未必能用单一 threshold 平滑实现同样的 eligible gate rate。`,
      ...data.notes.map(item => item),
    ].map(item => `<li>${{item}}</li>`).join("");
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    records_path = Path(args.records_csv)
    source_meta: Dict[str, Any] = {}
    if args.source_json:
        source_path = Path(args.source_json)
        source_meta = json.loads(source_path.read_text(encoding="utf-8"))
        source_meta["_source_json"] = str(source_path.resolve())
    source_meta["_source_csv"] = str(records_path.resolve())

    report = build_report(
        load_records(args.records_csv),
        source_meta,
        [float(v) for v in args.budget_rates],
        include_variants=args.include_variants,
        page_title=str(args.page_title),
        page_badge=str(args.page_badge),
    )

    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_html.write_text(render_html(report), encoding="utf-8")
    print(output_json)
    print(output_html)


if __name__ == "__main__":
    main()
