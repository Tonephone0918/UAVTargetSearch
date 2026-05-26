from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY_ORDER = [
    "baseline_v1",
    "ablation_hist0",
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

SCAN_VARIANTS = [
    "baseline_v1",
    "ablation_hist0",
    "v3_hybrid_clear_fragility",
    "v_next_prop_gap_support_region",
    "v_next2_prop_fragility_support_region",
    "cand_clearmin_prop_fragility_support_region",
    "cand_clearmin_prop_support_region",
    "vnext_tune_proxy_prop_gap_region",
    "vnext_tune_exact_prop_support",
    "vnext_tune_exact_prop_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a static HTML dashboard for offline risk validation results.")
    parser.add_argument("--input-json", required=True, help="Offline validation JSON path.")
    parser.add_argument("--output-html", required=True, help="Output HTML path.")
    parser.add_argument(
        "--include-variants",
        type=str,
        nargs="*",
        default=None,
        help="Optional variant-name allowlist. When set, only these variants are shown in the dashboard.",
    )
    parser.add_argument(
        "--focus-variant",
        type=str,
        default=None,
        help="Optional variant used by the component panel. Defaults to v_next, then v_next2, then the first shown variant.",
    )
    parser.add_argument("--page-title", type=str, default="Fixed Safe Trajectory Offline Validation")
    parser.add_argument("--page-badge", type=str, default="Fixed Safe Trajectory Offline Validation")
    return parser.parse_args()


def build_dashboard_payload(
    raw: dict,
    *,
    include_variants: list[str] | None = None,
    focus_variant: str | None = None,
    page_title: str = "Fixed Safe Trajectory Offline Validation",
    page_badge: str = "Fixed Safe Trajectory Offline Validation",
) -> dict:
    variants = {item["variant"]: item for item in raw["variant_summaries"]}
    allowed = {str(name) for name in include_variants} if include_variants else None
    selected = []
    for name in DISPLAY_ORDER:
        if name not in variants:
            continue
        if allowed is not None and name not in allowed:
            continue
        item = variants[name]
        best = item["best_threshold_metrics"]
        best_eligible = item["best_threshold_eligible_metrics"]
        meta = DISPLAY_META[name]
        selected.append(
            {
                "name": name,
                "label": meta["label"],
                "short": meta["short"],
                "formula": meta["formula"],
                "best_threshold_all": item["best_threshold_by_precision_plus_recall"],
                "best_threshold_primary": item["best_threshold_by_eligible_precision_plus_recall"],
                "all_gate_rate": best["gate_rate"],
                "primary_gate_rate": best_eligible["eligible_gate_rate"],
                "wasted_gate_rate": best_eligible["wasted_gate_rate"],
                "all_precision": best["precision_need_rec"],
                "primary_precision": best_eligible["eligible_precision_need_rec"],
                "primary_recall": best_eligible["recall_need_rec"],
                "all_score_pos": item["risk_score_need_rec_pos_mean"],
                "all_score_neg": item["risk_score_need_rec_neg_mean"],
                "all_score_delta": item["risk_score_need_rec_pos_mean"] - item["risk_score_need_rec_neg_mean"],
                "primary_score_pos": item["eligible_risk_score_need_rec_pos_mean"],
                "primary_score_neg": item["eligible_risk_score_need_rec_neg_mean"],
                "primary_score_delta": item["eligible_risk_score_need_rec_pos_mean"] - item["eligible_risk_score_need_rec_neg_mean"],
                "hard_empty_rate": item.get("dead_end_hard_agent_step_rate", item["hard_empty_agent_step_rate"]),
                "ineligible_nonempty_rate": item["ineligible_nonempty_agent_step_rate"],
                "threshold_scan": item["threshold_scan"],
            }
        )

    if not selected:
        raise ValueError("No variants selected for dashboard rendering.")

    shown_names = {item["name"] for item in selected}
    if focus_variant and focus_variant in shown_names:
        focus_name = focus_variant
    elif "v_next_prop_gap_support_region" in shown_names:
        focus_name = "v_next_prop_gap_support_region"
    elif "v_next2_prop_fragility_support_region" in shown_names:
        focus_name = "v_next2_prop_fragility_support_region"
    else:
        focus_name = selected[0]["name"]
    focus = variants[focus_name]
    component_rows = []
    component_specs = [
        ("score", "eligible_risk_score_need_rec_pos_mean", "eligible_risk_score_need_rec_neg_mean"),
        ("prop_clear", "eligible_risk_clear_prop_need_rec_pos_mean", "eligible_risk_clear_prop_need_rec_neg_mean"),
        ("support", "eligible_risk_support_need_rec_pos_mean", "eligible_risk_support_need_rec_neg_mean"),
        ("region", "eligible_risk_region_need_rec_pos_mean", "eligible_risk_region_need_rec_neg_mean"),
    ]
    if focus_name == "v_next2_prop_fragility_support_region":
        component_specs.insert(2, ("fragility", "eligible_risk_fragility_need_rec_pos_mean", "eligible_risk_fragility_need_rec_neg_mean"))
    else:
        component_specs.insert(2, ("clear_gap", "eligible_risk_clear_gap_need_rec_pos_mean", "eligible_risk_clear_gap_need_rec_neg_mean"))

    for label, pos_key, neg_key in component_specs:
        pos = float(focus[pos_key])
        neg = float(focus[neg_key])
        component_rows.append(
            {
                "label": label,
                "pos": pos,
                "neg": neg,
                "delta": pos - neg,
            }
        )

    trajectory = raw["trajectory_summary"]
    return {
        "checkpoint": raw["checkpoint"],
        "episodes": raw["episodes"],
        "env_steps": raw["env_steps"],
        "page_title": str(page_title),
        "page_badge": str(page_badge),
        "primary_view": raw.get("validation_primary_view", "eligible_only"),
        "trajectory_summary": trajectory,
        "thresholds": raw["thresholds"],
        "variants": selected,
        "scan_variants": [item for item in selected if item["name"] in SCAN_VARIANTS],
        "focus_variant": {
            "name": focus_name,
            "short": DISPLAY_META[focus_name]["short"],
            "formula": DISPLAY_META[focus_name]["formula"],
        },
        "vnext_components": component_rows,
        "notes": raw["notes"],
    }


def render_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Offline Safe Validation Risk Variants</title>
  <style>
    :root {{
      --bg: #f7f2e8;
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
        radial-gradient(circle at top left, rgba(177, 77, 47, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(41, 95, 138, 0.12), transparent 25%),
        linear-gradient(180deg, #fbf6ee 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1240px; margin: 0 auto; padding: 28px 20px 48px; }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 28px; }}
    .panel {{ padding: 20px; }}
    h1, h2 {{ margin: 0; }}
    h1 {{ font-size: 34px; }}
    h2 {{ font-size: 20px; margin-bottom: 14px; }}
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
    .meta, .cards {{
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
    .box.highlight {{
      background: linear-gradient(180deg, rgba(47,125,109,0.10), rgba(255,255,255,0.8));
      border-color: rgba(47,125,109,0.22);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 18px;
      margin-top: 20px;
    }}
    .span-12 {{ grid-column: span 12; }}
    .span-6 {{ grid-column: span 6; }}
    .span-4 {{ grid-column: span 4; }}
    @media (max-width: 960px) {{
      .span-6, .span-4 {{ grid-column: span 12; }}
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
    .axis-note {{
      margin: -4px 0 12px;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.55;
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
    .paths {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
      font-size: 14px;
    }}
    .path-row {{
      padding: 12px 14px;
      background: rgba(31, 41, 51, 0.04);
      border-radius: 14px;
      border: 1px solid rgba(82, 96, 109, 0.1);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <span class="badge">{payload["page_badge"]}</span>
      <h1 style="margin-top: 12px;">{payload["page_title"]}</h1>
      <p style="margin-top: 10px;">
        这页展示的是同一条 <code>safe</code> 轨迹上的离线打标结果。所有 variant 共用同一批
        <code>agent-step</code> 样本和同一套 <code>A_rec_oracle</code> 标签，因此更适合看
        “risk 本身到底有没有把 <code>eligible</code> 样本里的 <code>need_rec</code> 排到前面”。
        主页口径已经切到 <code>eligible-only</code>，同时把 <code>A_hard empty (dead_end_hard)</code>
        和 <code>ineligible but nonempty A_hard</code> 单独分桶展示。
      </p>
      <div class="meta" id="hero-meta"></div>
    </section>

    <div class="grid">
      <section class="panel span-12">
        <h2>快速判断</h2>
        <div class="cards" id="summary-cards"></div>
      </section>

      <section class="panel span-12">
        <h2>Best Threshold 对比</h2>
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              <th>Formula</th>
              <th>Best Th (All)</th>
              <th>Best Th (Primary)</th>
              <th>All Gate Rate</th>
              <th>Primary Gate Rate</th>
              <th>dead_end_hard</th>
              <th>ineligible_nonempty</th>
              <th>Wasted Gate Rate</th>
              <th>All Precision</th>
              <th>Primary Precision</th>
              <th>Recall</th>
              <th>eligible score_pos</th>
              <th>eligible score_neg</th>
              <th>eligible delta</th>
            </tr>
          </thead>
          <tbody id="variant-body"></tbody>
        </table>
      </section>

      <section class="panel span-6">
        <h2>Recall Threshold Scan</h2>
        <p class="axis-note">横轴：风险门限 <code>eta</code>，无量纲。纵轴：Recall，单位为 %，表示被 gate 命中的 <code>need_rec</code> 样本占全部 <code>need_rec</code> 样本的比例。</p>
        <div class="chart" id="recall-chart"></div>
        <div class="legend" id="scan-legend-a"></div>
      </section>

      <section class="panel span-6">
        <h2>Eligible Precision Threshold Scan</h2>
        <p class="axis-note">横轴：风险门限 <code>eta</code>，无量纲。纵轴：Eligible Precision，单位为 %，表示已触发 gate 且属于 <code>eligible</code> 的样本中，真正 <code>need_rec</code> 的比例。</p>
        <div class="chart" id="eligible-chart"></div>
        <div class="legend" id="scan-legend-b"></div>
      </section>

      <section class="panel span-6">
        <h2>Primary Gate Rate Threshold Scan</h2>
        <p class="axis-note">横轴：风险门限 <code>eta</code>，无量纲。纵轴：Eligible Gate Rate，单位为 %，表示全部 <code>eligible</code> 样本里，有多少被送去做 <code>A_rec</code> 检查。</p>
        <div class="chart" id="gate-chart"></div>
        <div class="legend" id="scan-legend-c"></div>
      </section>

      <section class="panel span-6">
        <h2 id="component-title">Focus Variant 分量均值差</h2>
        <div class="chart" id="component-chart"></div>
        <p style="margin-top: 10px;">
          这里的条形值是 <code>mean(score | need_rec=1, eligible=1) - mean(score | need_rec=0, eligible=1)</code>。理想情况下应为正。
        </p>
      </section>

      <section class="panel span-12">
        <h2 id="component-table-title">Focus Variant 分量细表</h2>
        <table>
          <thead>
            <tr>
              <th>Component</th>
              <th>pos mean</th>
              <th>neg mean</th>
              <th>delta</th>
            </tr>
          </thead>
          <tbody id="component-body"></tbody>
        </table>
      </section>

      <section class="panel span-4">
        <h2>这次结果怎么读</h2>
        <ul id="takeaways"></ul>
      </section>

      <section class="panel span-4">
        <h2>原始结果文件</h2>
        <div class="paths" id="paths"></div>
      </section>

      <section class="panel span-4">
        <h2>脚本备注</h2>
        <ul id="notes"></ul>
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
    const fmt = value => Number(value).toFixed(3);
    const short = name => data.variants.find(v => v.name === name).short;

    const trajectory = data.trajectory_summary;
    document.getElementById("hero-meta").innerHTML = [
      `<div class="box"><strong>Checkpoint</strong>${{data.checkpoint}}</div>`,
      `<div class="box"><strong>Trajectory</strong>${{data.episodes}} safe episodes, ${{data.env_steps}} env steps</div>`,
      `<div class="box"><strong>Need Rec Rate</strong>${{pct(trajectory.need_rec_rate)}}</div>`,
      `<div class="box"><strong>Eligible Rate</strong>${{pct(trajectory.eligible_agent_step_rate)}} proposed actions survive A_hard</div>`,
      `<div class="box"><strong>dead_end_hard / A_hard empty</strong>${{pct(trajectory.dead_end_hard_agent_step_rate || trajectory.hard_empty_agent_step_rate || 0)}}</div>`,
      `<div class="box"><strong>Ineligible, A_hard nonempty</strong>${{pct(trajectory.ineligible_nonempty_agent_step_rate || 0)}}</div>`,
    ].join("");

    const focusVariant = data.variants.find(v => v.name === data.focus_variant.name);
    const vnext = data.variants.find(v => v.name === "v_next_prop_gap_support_region");
    const bestPrecisionVariant = data.variants.reduce((best, item) => !best || item.primary_precision > best.primary_precision ? item : best, null);
    const bestRecallVariant = data.variants.reduce((best, item) => !best || item.primary_recall > best.primary_recall ? item : best, null);
    const secondVariant = data.variants.find(v => v.name !== focusVariant.name) || focusVariant;
    document.getElementById("component-title").textContent = `${{data.focus_variant.short}} 分量均值差`;
    document.getElementById("component-table-title").textContent = `${{data.focus_variant.short}} 分量细表`;
    document.getElementById("summary-cards").innerHTML = [
      {{
        title: "Primary View",
        value: data.primary_view,
        sub: `主验证口径只看 eligible 样本，即 proposed_action 已经在 A_hard 内的 agent-step`,
        highlight: true,
      }},
      {{
        title: `${{data.focus_variant.short}} recall`,
        value: pct(focusVariant.primary_recall),
        sub: `best threshold = ${{focusVariant.best_threshold_primary.toFixed(2)}}，当前主观察对象的 recall`,
        highlight: false,
      }},
      {{
        title: `${{data.focus_variant.short}} eligible precision`,
        value: pct(focusVariant.primary_precision),
        sub: `用来判断 risk 把 eligible 的 need_rec 排到前面的能力`,
        highlight: false,
      }},
      {{
        title: `${{data.focus_variant.short}} wasted gate`,
        value: pct(focusVariant.wasted_gate_rate),
        sub: `越低越好，表示 gate 没有浪费在非 eligible 样本上的比例`,
        highlight: false,
      }},
      {{
        title: "dead_end_hard bucket",
        value: pct(trajectory.dead_end_hard_agent_step_rate || trajectory.hard_empty_agent_step_rate || 0),
        sub: `这部分样本不属于 A_hard -> A_rec gate 判别问题，应与 eligible 样本分开看`,
        highlight: false,
      }},
      {{
        title: "Best Precision",
        value: `${{bestPrecisionVariant.short}} / ${{pct(bestPrecisionVariant.primary_precision)}}`,
        sub: `当前展示 variants 中，eligible precision 最好的版本`,
        highlight: false,
      }},
      {{
        title: "Best Recall",
        value: `${{bestRecallVariant.short}} / ${{pct(bestRecallVariant.primary_recall)}}`,
        sub: `当前展示 variants 中，recall 最好的版本`,
        highlight: false,
      }},
      {{
        title: secondVariant.name === focusVariant.name ? "Need Rec Rate" : `${{secondVariant.short}} 参考线`,
        value: secondVariant.name === focusVariant.name ? pct(trajectory.need_rec_rate) : pct(secondVariant.primary_precision),
        sub: secondVariant.name === focusVariant.name
          ? `固定 safe 轨迹上的 oracle need_rec 基础比例`
          : `${{secondVariant.short}} recall = ${{pct(secondVariant.primary_recall)}}，方便和 ${{focusVariant.short}} 直接比较`,
        highlight: false,
      }},
    ].map(card => `
      <div class="box ${{card.highlight ? "highlight" : ""}}">
        <strong>${{card.title}}</strong>
        <div class="value">${{card.value}}</div>
        <div class="sub">${{card.sub}}</div>
      </div>
    `).join("");

    document.getElementById("variant-body").innerHTML = data.variants.map(item => {{
      const bestClass = item.name === data.focus_variant.name ? "best" : (item.primary_score_delta < 0 ? "warn" : "");
      return `
        <tr class="${{bestClass}}">
          <td><strong>${{item.label}}</strong></td>
          <td>${{item.formula}}</td>
          <td>${{item.best_threshold_all.toFixed(2)}}</td>
          <td>${{item.best_threshold_primary.toFixed(2)}}</td>
          <td>${{pct(item.all_gate_rate)}}</td>
          <td>${{pct(item.primary_gate_rate)}}</td>
          <td>${{pct(item.hard_empty_rate)}}</td>
          <td>${{pct(item.ineligible_nonempty_rate)}}</td>
          <td>${{pct(item.wasted_gate_rate)}}</td>
          <td>${{pct(item.all_precision)}}</td>
          <td>${{pct(item.primary_precision)}}</td>
          <td>${{pct(item.primary_recall)}}</td>
          <td>${{fmt(item.primary_score_pos)}}</td>
          <td>${{fmt(item.primary_score_neg)}}</td>
          <td>${{item.primary_score_delta >= 0 ? "+" : ""}}${{fmt(item.primary_score_delta)}}</td>
        </tr>
      `;
    }}).join("");

    document.getElementById("component-body").innerHTML = data.vnext_components.map(row => `
      <tr class="${{row.delta < 0 ? "warn" : ""}}">
        <td><strong>${{row.label}}</strong></td>
        <td>${{fmt(row.pos)}}</td>
        <td>${{fmt(row.neg)}}</td>
        <td>${{row.delta >= 0 ? "+" : ""}}${{fmt(row.delta)}}</td>
      </tr>
    `).join("");

    document.getElementById("takeaways").innerHTML = [
      `主验证口径现在是 eligible-only，因为 need_rec 只对 proposed_action 已在 A_hard 内的样本有语义。`,
      `dead_end_hard / A_hard empty 与 ineligible-but-nonempty 已单独分桶；它们安全上重要，但不应混入 A_hard -> A_rec gate 排序判断。`,
      `在 eligible-only 口径下，${{data.focus_variant.short}} 的 score_pos = ${{fmt(focusVariant.primary_score_pos)}}，score_neg = ${{fmt(focusVariant.primary_score_neg)}}，delta = ${{focusVariant.primary_score_delta >= 0 ? "+" : ""}}${{fmt(focusVariant.primary_score_delta)}}。`,
      `${{data.focus_variant.short}} 的 recall = ${{pct(focusVariant.primary_recall)}}，eligible precision = ${{pct(focusVariant.primary_precision)}}；需要同时结合 best precision / best recall 版本一起判断。`,
      `${{bestPrecisionVariant.short}} 当前拿到最高 eligible precision = ${{pct(bestPrecisionVariant.primary_precision)}}；${{bestRecallVariant.short}} 当前拿到最高 recall = ${{pct(bestRecallVariant.primary_recall)}}。`,
      `${{vnext ? ("v_next 参考值：precision = " + pct(vnext.primary_precision) + "，recall = " + pct(vnext.primary_recall) + "。") : "本页未包含 v_next 参考线。"}}`,
    ].map(item => `<li>${{item}}</li>`).join("");

    document.getElementById("paths").innerHTML = [
      `<div class="path-row"><strong>主 JSON</strong><br /><code>${{data.input_json || ""}}</code></div>`,
      `<div class="path-row"><strong>逐样本 CSV</strong><br /><code>${{data.records_csv || ""}}</code></div>`,
      `<div class="path-row"><strong>页面本身</strong><br /><code>${{data.output_html || ""}}</code></div>`,
    ].join("");

    document.getElementById("notes").innerHTML = data.notes.map(note => `<li>${{note}}</li>`).join("");

    function renderLegend(elId, items) {{
      document.getElementById(elId).innerHTML = items.map(item => `
        <span class="legend-item"><span class="dot" style="background:${{item.color}};"></span>${{item.label}}</span>
      `).join("");
    }}

    function renderLineChart(el, series, labels, maxValueOverride = null) {{
      const width = 680;
      const height = 260;
      const pad = {{ top: 18, right: 18, bottom: 34, left: 42 }};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const allValues = series.flatMap(s => s.values);
      const maxVal = maxValueOverride || Math.max(...allValues, 1e-6);
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
      el.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="chart">${{axes}}${{ticks}}${{xLabels}}${{paths}}</svg>`;
    }}

    function renderBarChart(el, rows) {{
      const width = 680;
      const height = 260;
      const pad = {{ top: 18, right: 24, bottom: 34, left: 80 }};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const maxAbs = Math.max(...rows.map(r => Math.abs(r.delta)), 0.001);
      const zeroX = pad.left + plotW / 2;
      const rowH = plotH / rows.length;
      const bars = rows.map((row, idx) => {{
        const y = pad.top + idx * rowH + rowH * 0.18;
        const h = rowH * 0.64;
        const w = (Math.abs(row.delta) / maxAbs) * (plotW / 2 - 10);
        const x = row.delta >= 0 ? zeroX : zeroX - w;
        const color = row.delta >= 0 ? "#2f7d6d" : "#b14d2f";
        return `
          <text x="${{pad.left - 8}}" y="${{y + h / 2 + 4}}" text-anchor="end" font-size="12" fill="#52606d">${{row.label}}</text>
          <rect x="${{x}}" y="${{y}}" width="${{w}}" height="${{h}}" rx="8" fill="${{color}}" />
          <text x="${{row.delta >= 0 ? x + w + 8 : x - 8}}" y="${{y + h / 2 + 4}}" text-anchor="${{row.delta >= 0 ? "start" : "end"}}" font-size="11" fill="#52606d">${{row.delta >= 0 ? "+" : ""}}${{row.delta.toFixed(3)}}</text>
        `;
      }}).join("");
      const axes = `
        <line x1="${{zeroX}}" y1="${{pad.top}}" x2="${{zeroX}}" y2="${{pad.top + plotH}}" stroke="#9fb3c8" stroke-width="1.2" />
        <text x="${{zeroX - 6}}" y="${{pad.top - 4}}" text-anchor="end" font-size="11" fill="#52606d">neg</text>
        <text x="${{zeroX + 6}}" y="${{pad.top - 4}}" text-anchor="start" font-size="11" fill="#52606d">pos</text>
      `;
      el.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="bar-chart">${{axes}}${{bars}}</svg>`;
    }}

    const labels = data.thresholds.map(v => v.toFixed(2));
    const recallSeries = data.scan_variants.map(item => ({{
      label: item.short,
      color: COLORS[item.name],
      values: item.threshold_scan.map(row => row.recall_need_rec),
    }}));
    const eligibleSeries = data.scan_variants.map(item => ({{
      label: item.short,
      color: COLORS[item.name],
      values: item.threshold_scan.map(row => row.eligible_precision_need_rec),
    }}));
    const gateSeries = data.scan_variants.map(item => ({{
      label: item.short,
      color: COLORS[item.name],
      values: item.threshold_scan.map(row => row.eligible_gate_rate),
    }}));

    renderLineChart(document.getElementById("recall-chart"), recallSeries, labels);
    renderLineChart(document.getElementById("eligible-chart"), eligibleSeries, labels);
    renderLineChart(document.getElementById("gate-chart"), gateSeries, labels);
    renderLegend("scan-legend-a", recallSeries);
    renderLegend("scan-legend-b", eligibleSeries);
    renderLegend("scan-legend-c", gateSeries);
    renderBarChart(document.getElementById("component-chart"), data.vnext_components);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_html)
    payload = build_dashboard_payload(
        json.loads(input_path.read_text(encoding="utf-8")),
        include_variants=args.include_variants,
        focus_variant=args.focus_variant,
        page_title=str(args.page_title),
        page_badge=str(args.page_badge),
    )
    payload["input_json"] = str(input_path.resolve())
    payload["output_html"] = str(output_path.resolve())
    payload["records_csv"] = str(input_path.with_name(f"{input_path.stem}_records.csv").resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(payload), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
