from __future__ import annotations

import csv
import html
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNS_BASE = ROOT / "runs" / "formal_progressive_seed_compare_20260426"
CKPTS_BASE = ROOT / "checkpoints" / "formal_progressive_seed_compare_20260426"
OUT_DIR = ROOT / "runs" / "progressive_mechanism_20260428"

METHODS = [
    "non_progressive",
    "threshold_only_progressive",
    "safeearly_progressive",
]
TARGET_METRICS = [
    "collision_count",
    "guarantee_broken_rate",
    "recursive_gate_rate",
    "episode_return",
    "avg_reward",
    "perf_shield_time_ms",
    "perf_recursive_time_ms",
    "dead_end_rec_rate",
    "search_rate",
]
FORMAL_METRICS = [
    "search_rate_mean",
    "collision_count_mean",
    "guarantee_broken_rate_mean",
    "dead_end_hard_rate_mean",
    "dead_end_rec_rate_mean",
    "recursive_gate_rate_mean",
    "perf_shield_time_ms_mean",
    "perf_recursive_time_ms_mean",
]


@dataclass(frozen=True)
class MethodSchedule:
    mode: str
    recursive_gate_mode: str
    risk_threshold: float
    lookahead_horizon: int
    progressive_enabled: bool
    progressive_early_mode: str
    progressive_early_end_ratio: float
    progressive_late_start_ratio: float
    progressive_early_risk_threshold: float
    progressive_mid_risk_threshold: float
    progressive_late_risk_threshold: float
    progressive_mid_lookahead_horizon: int
    progressive_late_lookahead_horizon: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    raw = str(raw).strip()
    if raw == "":
        return None
    return float(raw)


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _safe_std(values: Iterable[float]) -> float:
    values = list(values)
    return pstdev(values) if len(values) > 1 else 0.0


def _load_schedule(method: str, training_seed: int = 1) -> MethodSchedule:
    ckpt_path = CKPTS_BASE / method / f"seed{training_seed}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    shield_cfg = ckpt["config"]["shield"]
    return MethodSchedule(
        mode=str(shield_cfg.get("mode", "recursive")),
        recursive_gate_mode=str(shield_cfg.get("recursive_gate_mode", "risk")),
        risk_threshold=float(shield_cfg.get("risk_threshold", 0.35)),
        lookahead_horizon=int(shield_cfg.get("lookahead_horizon", 1)),
        progressive_enabled=bool(shield_cfg.get("progressive_enabled", False)),
        progressive_early_mode=str(shield_cfg.get("progressive_early_mode", "safe")),
        progressive_early_end_ratio=float(shield_cfg.get("progressive_early_end_ratio", 0.33)),
        progressive_late_start_ratio=float(shield_cfg.get("progressive_late_start_ratio", 0.67)),
        progressive_early_risk_threshold=float(shield_cfg.get("progressive_early_risk_threshold", 0.90)),
        progressive_mid_risk_threshold=float(shield_cfg.get("progressive_mid_risk_threshold", 0.35)),
        progressive_late_risk_threshold=float(shield_cfg.get("progressive_late_risk_threshold", 0.35)),
        progressive_mid_lookahead_horizon=int(shield_cfg.get("progressive_mid_lookahead_horizon", 1)),
        progressive_late_lookahead_horizon=int(shield_cfg.get("progressive_late_lookahead_horizon", 1)),
    )


def _derived_mode(schedule: MethodSchedule, stage: str) -> str:
    if not schedule.progressive_enabled or stage == "fixed":
        return schedule.mode
    if stage == "early":
        return schedule.progressive_early_mode
    return "recursive"


def _derived_threshold(schedule: MethodSchedule, stage: str, recorded: float | None) -> float:
    if recorded is not None:
        return float(recorded)
    if not schedule.progressive_enabled or stage == "fixed":
        return schedule.risk_threshold
    if stage == "early":
        return schedule.progressive_early_risk_threshold
    if stage == "mid":
        return schedule.progressive_mid_risk_threshold
    return schedule.progressive_late_risk_threshold


def _derived_horizon(schedule: MethodSchedule, stage: str, recorded: float | None) -> int:
    if recorded is not None:
        return int(round(float(recorded)))
    if not schedule.progressive_enabled or stage == "fixed":
        return schedule.lookahead_horizon
    if stage == "early":
        return 1
    if stage == "mid":
        return schedule.progressive_mid_lookahead_horizon
    return schedule.progressive_late_lookahead_horizon


def _inventory_rows() -> tuple[list[dict[str, object]], dict[str, MethodSchedule]]:
    rows: list[dict[str, object]] = []
    schedules: dict[str, MethodSchedule] = {}
    for method in METHODS:
        schedule = _load_schedule(method)
        schedules[method] = schedule
        for seed_dir in sorted((RUNS_BASE / method).glob("seed*")):
            training_seed = int(seed_dir.name.replace("seed", ""))
            metrics_path = seed_dir / "metrics_summary.csv"
            event_paths = sorted(seed_dir.glob("events.out.tfevents.*"))
            metrics_rows = _read_csv(metrics_path)
            split_counts = Counter(row["split"] for row in metrics_rows)
            stage_counts = Counter(row.get("progressive_stage", "fixed") for row in metrics_rows)
            rows.append(
                {
                    "method": method,
                    "training_seed": training_seed,
                    "run_dir": str(seed_dir.relative_to(ROOT)),
                    "checkpoint": str((CKPTS_BASE / method / seed_dir.name / "best.pt").relative_to(ROOT)),
                    "metrics_summary": str(metrics_path.relative_to(ROOT)),
                    "events_count": len(event_paths),
                    "events_example": str(event_paths[0].relative_to(ROOT)) if event_paths else "",
                    "metrics_rows": len(metrics_rows),
                    "train_rows": split_counts.get("train", 0),
                    "eval_rows": split_counts.get("eval", 0),
                    "stage_counts": "; ".join(f"{k}:{v}" for k, v in sorted(stage_counts.items())),
                    "progressive_enabled": int(schedule.progressive_enabled),
                    "default_runtime_mode": schedule.mode,
                    "default_runtime_horizon": schedule.lookahead_horizon,
                    "default_runtime_threshold": schedule.risk_threshold,
                }
            )
    return rows, schedules


def _extract_training_curves(schedules: dict[str, MethodSchedule]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_seed_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for method in METHODS:
        schedule = schedules[method]
        for seed_dir in sorted((RUNS_BASE / method).glob("seed*")):
            training_seed = int(seed_dir.name.replace("seed", ""))
            for row in _read_csv(seed_dir / "metrics_summary.csv"):
                split = row["split"]
                epoch = int(row["epoch"])
                stage = row.get("progressive_stage", "fixed") or "fixed"
                progress = _to_float(row.get("progressive_progress")) or 0.0
                recorded_h = _to_float(row.get("effective_lookahead_horizon"))
                recorded_eta = _to_float(row.get("effective_risk_threshold"))
                out = {
                    "row_type": "seed",
                    "model": method,
                    "training_seed": training_seed,
                    "split": split,
                    "epoch": epoch,
                    "phase": row.get("phase", ""),
                    "progressive_enabled": int(schedule.progressive_enabled),
                    "progressive_stage": stage,
                    "progressive_progress": progress,
                    "effective_shield_mode": _derived_mode(schedule, stage),
                    "effective_lookahead_horizon": _derived_horizon(schedule, stage, recorded_h),
                    "effective_risk_threshold": _derived_threshold(schedule, stage, recorded_eta),
                }
                for metric in TARGET_METRICS:
                    out[metric] = _to_float(row.get(metric))
                per_seed_rows.append(out)
                grouped[(method, split, epoch)].append(out)

    aggregate_rows: list[dict[str, object]] = []
    for (method, split, epoch), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        stages = Counter(str(row["progressive_stage"]) for row in rows)
        modes = Counter(str(row["effective_shield_mode"]) for row in rows)
        out = {
            "row_type": "aggregate",
            "model": method,
            "training_seed": "all",
            "split": split,
            "epoch": epoch,
            "phase": rows[0]["phase"],
            "progressive_enabled": rows[0]["progressive_enabled"],
            "progressive_stage": stages.most_common(1)[0][0],
            "progressive_stage_consensus": stages.most_common(1)[0][0],
            "progressive_progress_mean": _safe_mean(float(row["progressive_progress"]) for row in rows),
            "progressive_progress_std": _safe_std(float(row["progressive_progress"]) for row in rows),
            "effective_shield_mode": modes.most_common(1)[0][0],
            "effective_lookahead_horizon_mean": _safe_mean(float(row["effective_lookahead_horizon"]) for row in rows),
            "effective_lookahead_horizon_std": _safe_std(float(row["effective_lookahead_horizon"]) for row in rows),
            "effective_risk_threshold_mean": _safe_mean(float(row["effective_risk_threshold"]) for row in rows),
            "effective_risk_threshold_std": _safe_std(float(row["effective_risk_threshold"]) for row in rows),
        }
        for metric in TARGET_METRICS:
            vals = [float(row[metric]) for row in rows if row[metric] is not None]
            out[f"{metric}_mean"] = _safe_mean(vals)
            out[f"{metric}_std"] = _safe_std(vals)
        aggregate_rows.append(out)
    return per_seed_rows, aggregate_rows


def _stage_rows(schedules: dict[str, MethodSchedule]) -> list[dict[str, object]]:
    seed_stage_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)

    for method in METHODS:
        schedule = schedules[method]
        for seed_dir in sorted((RUNS_BASE / method).glob("seed*")):
            training_seed = int(seed_dir.name.replace("seed", ""))
            by_stage: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            for row in _read_csv(seed_dir / "metrics_summary.csv"):
                by_stage[(row["split"], row.get("progressive_stage", "fixed") or "fixed")].append(row)
            for (split, stage), rows in sorted(by_stage.items()):
                recorded_h_vals = [_to_float(r.get("effective_lookahead_horizon")) for r in rows]
                recorded_eta_vals = [_to_float(r.get("effective_risk_threshold")) for r in rows]
                out = {
                    "row_type": "seed",
                    "model": method,
                    "training_seed": training_seed,
                    "split": split,
                    "progressive_stage": stage,
                    "stage_epoch_count": len(rows),
                    "stage_epoch_min": min(int(r["epoch"]) for r in rows),
                    "stage_epoch_max": max(int(r["epoch"]) for r in rows),
                    "stage_progress_min": min(_to_float(r.get("progressive_progress")) or 0.0 for r in rows),
                    "stage_progress_max": max(_to_float(r.get("progressive_progress")) or 0.0 for r in rows),
                    "effective_shield_mode": _derived_mode(schedule, stage),
                    "effective_lookahead_horizon": _derived_horizon(schedule, stage, _safe_mean(v for v in recorded_h_vals if v is not None) if any(v is not None for v in recorded_h_vals) else None),
                    "effective_risk_threshold": _derived_threshold(schedule, stage, _safe_mean(v for v in recorded_eta_vals if v is not None) if any(v is not None for v in recorded_eta_vals) else None),
                }
                for metric in TARGET_METRICS:
                    vals = [_to_float(r.get(metric)) for r in rows]
                    vals = [v for v in vals if v is not None]
                    out[metric] = _safe_mean(vals)
                seed_stage_rows.append(out)
                grouped[(method, split, stage)].append(out)

    aggregate_rows: list[dict[str, object]] = []
    for (method, split, stage), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        out = {
            "row_type": "aggregate",
            "model": method,
            "training_seed": "all",
            "split": split,
            "progressive_stage": stage,
            "stage_epoch_count_mean": _safe_mean(float(r["stage_epoch_count"]) for r in rows),
            "stage_epoch_count_std": _safe_std(float(r["stage_epoch_count"]) for r in rows),
            "stage_epoch_min_mean": _safe_mean(float(r["stage_epoch_min"]) for r in rows),
            "stage_epoch_max_mean": _safe_mean(float(r["stage_epoch_max"]) for r in rows),
            "stage_progress_min_mean": _safe_mean(float(r["stage_progress_min"]) for r in rows),
            "stage_progress_max_mean": _safe_mean(float(r["stage_progress_max"]) for r in rows),
            "effective_shield_mode": Counter(str(r["effective_shield_mode"]) for r in rows).most_common(1)[0][0],
            "effective_lookahead_horizon_mean": _safe_mean(float(r["effective_lookahead_horizon"]) for r in rows),
            "effective_lookahead_horizon_std": _safe_std(float(r["effective_lookahead_horizon"]) for r in rows),
            "effective_risk_threshold_mean": _safe_mean(float(r["effective_risk_threshold"]) for r in rows),
            "effective_risk_threshold_std": _safe_std(float(r["effective_risk_threshold"]) for r in rows),
        }
        for metric in TARGET_METRICS:
            out[f"{metric}_mean"] = _safe_mean(float(r[metric]) for r in rows)
            out[f"{metric}_std"] = _safe_std(float(r[metric]) for r in rows)
        aggregate_rows.append(out)
    return seed_stage_rows + aggregate_rows


def _summary_rows(schedules: dict[str, MethodSchedule]) -> list[dict[str, object]]:
    raw_per_seed_path = RUNS_BASE / "formal_compare_multiseed5x5_raw" / "per_seed_metrics.csv"
    raw_rows = _read_csv(raw_per_seed_path)
    formal_rows: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        model_name = row["model"].rsplit("_seed", 1)[0]
        grouped[model_name].append(row)
    for method, rows in grouped.items():
        out: dict[str, object] = {
            "model": method,
            "training_seed_count": len({row["model"] for row in rows}),
            "eval_seed_count_per_checkpoint": len({row["seed"] for row in rows if row["model"].endswith("_seed1")}),
        }
        metric_names = [key for key in rows[0].keys() if key not in {"model", "checkpoint", "seed", "episodes"}]
        for metric in metric_names:
            vals = [_to_float(row.get(metric)) for row in rows]
            vals = [v for v in vals if v is not None]
            out[f"{metric}_mean"] = _safe_mean(vals)
            out[f"{metric}_std"] = _safe_std(vals)
        formal_rows[method] = out
    stage_path = OUT_DIR / "stage_metrics.csv"
    stage_rows = _read_csv(stage_path) if stage_path.exists() else []
    stage_lookup = {
        (row["model"], row["split"], row["progressive_stage"]): row
        for row in stage_rows
        if row.get("row_type") == "aggregate"
    }
    out_rows: list[dict[str, object]] = []
    for method in METHODS:
        schedule = schedules[method]
        row = formal_rows[method]
        out = {
            "model": method,
            "training_seed_count": int(row["training_seed_count"]),
            "eval_seed_count_per_checkpoint": int(row["eval_seed_count_per_checkpoint"]),
            "default_runtime_mode": schedule.mode,
            "default_runtime_horizon": schedule.lookahead_horizon,
            "default_runtime_threshold": schedule.risk_threshold,
            "progressive_enabled": int(schedule.progressive_enabled),
            "progressive_early_mode": schedule.progressive_early_mode,
            "progressive_early_end_ratio": schedule.progressive_early_end_ratio,
            "progressive_late_start_ratio": schedule.progressive_late_start_ratio,
            "progressive_early_risk_threshold": schedule.progressive_early_risk_threshold,
            "progressive_mid_risk_threshold": schedule.progressive_mid_risk_threshold,
            "progressive_late_risk_threshold": schedule.progressive_late_risk_threshold,
            "progressive_mid_lookahead_horizon": schedule.progressive_mid_lookahead_horizon,
            "progressive_late_lookahead_horizon": schedule.progressive_late_lookahead_horizon,
        }
        for key in FORMAL_METRICS:
            out[key] = _to_float(row.get(key))
        for stage in ["early", "mid", "late", "fixed"]:
            agg = stage_lookup.get((method, "eval", stage))
            if not agg:
                continue
            for field in [
                "collision_count_mean",
                "guarantee_broken_rate_mean",
                "recursive_gate_rate_mean",
                "episode_return_mean",
                "perf_shield_time_ms_mean",
                "dead_end_rec_rate_mean",
            ]:
                out[f"eval_{stage}_{field}"] = _to_float(agg.get(field))
        out_rows.append(out)
    return out_rows


def _write_inventory_md(inventory_rows: list[dict[str, object]], schedules: dict[str, MethodSchedule]) -> None:
    lines = [
        "# Asset Inventory",
        "",
        "Authoritative sources selected for this mechanism pass:",
        "",
        "- Training-process / stage analysis: `runs/formal_progressive_seed_compare_20260426/*/seed*/metrics_summary.csv`",
        "- Final default-formal comparison: `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv` and per-checkpoint companions",
        "- Schedule metadata backfill: checkpoint-embedded shield config from `checkpoints/formal_progressive_seed_compare_20260426/*/seed*/best.pt`",
        "- TensorBoard event files exist for all seeds, but were not used as the primary source because the CSV summaries already expose the needed metrics in a more stable tabular form.",
        "",
        "Consistency check:",
        "",
        "- All three target methods have 3 training seeds, matching `best.pt`, `metrics_summary.csv`, and TensorBoard event files.",
        "- `formal_compare_multiseed5x5_raw/per_seed_metrics.csv` is used as the default final-performance source, because it is the most direct unified 5x5 evaluation table.",
        "- The older precomputed `formal_compare_multiseed5x5/summary_metrics.csv` is retained as an asset, but some perf aggregates do not match the raw per-seed table; this pass therefore re-aggregates from raw instead of trusting the older summary verbatim.",
        "- The newer `formal_compare_with_dual_multiseed5x5` exists, but dual is intentionally out of scope for this pass.",
        "",
        "Per-run inventory:",
        "",
        "| method | training_seed | metrics_rows | train_rows | eval_rows | events_count | checkpoint | stage_counts |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in inventory_rows:
        lines.append(
            "| {method} | {training_seed} | {metrics_rows} | {train_rows} | {eval_rows} | {events_count} | `{checkpoint}` | {stage_counts} |".format(
                **row
            )
        )
    lines += [
        "",
        "Checkpoint schedule metadata (seed1 representative, verified from checkpoint config):",
        "",
        "| method | progressive_enabled | early_mode | early_eta | mid_H | mid_eta | late_H | late_eta |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        s = schedules[method]
        lines.append(
            f"| {method} | {int(s.progressive_enabled)} | {s.progressive_early_mode} | {s.progressive_early_risk_threshold:.2f} | {s.progressive_mid_lookahead_horizon} | {s.progressive_mid_risk_threshold:.2f} | {s.progressive_late_lookahead_horizon} | {s.progressive_late_risk_threshold:.2f} |"
        )
    lines += [
        "",
        "Immediate conclusions:",
        "",
        "- `threshold_only_progressive` and `safeearly_progressive` share the same early and mid schedule; the key schedule difference is late stage (`H=1, eta=0.35` vs `H=2, eta=0.55`).",
        "- `metrics_summary.csv` already contains `progressive_stage`, `effective_lookahead_horizon`, and `effective_risk_threshold`, so stage-level behavior can be recovered directly.",
        "- `effective_shield_mode` is not present in these older CSVs, so it is deterministically derived from the checkpoint schedule plus `progressive_stage`.",
        "",
    ]
    (OUT_DIR / "asset_inventory.md").write_text("\n".join(lines), encoding="utf-8")


def _write_simple_html(summary_rows: list[dict[str, object]], stage_rows: list[dict[str, object]]) -> None:
    stage_eval_rows = [
        row for row in stage_rows if row.get("row_type") == "aggregate" and row.get("split") == "eval"
    ]

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    summary_headers = [
        "model",
        "search_rate_mean",
        "collision_count_mean",
        "guarantee_broken_rate_mean",
        "dead_end_rec_rate_mean",
        "recursive_gate_rate_mean",
        "perf_shield_time_ms_mean",
        "perf_recursive_time_ms_mean",
    ]
    stage_headers = [
        "model",
        "progressive_stage",
        "effective_shield_mode",
        "effective_lookahead_horizon_mean",
        "effective_risk_threshold_mean",
        "collision_count_mean",
        "guarantee_broken_rate_mean",
        "recursive_gate_rate_mean",
        "episode_return_mean",
        "perf_shield_time_ms_mean",
    ]

    def table(headers: list[str], rows: list[dict[str, object]]) -> str:
        head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        body_parts = []
        for row in rows:
            cells = "".join(f"<td>{html.escape(fmt(row.get(h, '')))}</td>" for h in headers)
            body_parts.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Progressive Mechanism 20260428</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; font-size: 13px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Progressive Mechanism 20260428</h1>
  <p>Primary sources: <code>metrics_summary.csv</code> for training/stage behavior, <code>formal_compare_multiseed5x5</code> for default-formal endpoint comparison, and checkpoint configs for schedule backfill.</p>
  <h2>Default Formal Summary</h2>
  {table(summary_headers, summary_rows)}
  <h2>Eval Stage Summary</h2>
  {table(stage_headers, stage_eval_rows)}
</body>
</html>
"""
    (OUT_DIR / "overview.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_rows, schedules = _inventory_rows()
    _write_inventory_md(inventory_rows, schedules)

    curve_seed_rows, curve_agg_rows = _extract_training_curves(schedules)
    curve_fields = [
        "row_type",
        "model",
        "training_seed",
        "split",
        "epoch",
        "phase",
        "progressive_enabled",
        "progressive_stage",
        "progressive_progress",
        "effective_shield_mode",
        "effective_lookahead_horizon",
        "effective_risk_threshold",
        *TARGET_METRICS,
    ]
    agg_fields = [
        "row_type",
        "model",
        "training_seed",
        "split",
        "epoch",
        "phase",
        "progressive_enabled",
        "progressive_stage",
        "progressive_stage_consensus",
        "progressive_progress_mean",
        "progressive_progress_std",
        "effective_shield_mode",
        "effective_lookahead_horizon_mean",
        "effective_lookahead_horizon_std",
        "effective_risk_threshold_mean",
        "effective_risk_threshold_std",
    ] + [field for metric in TARGET_METRICS for field in (f"{metric}_mean", f"{metric}_std")]
    _write_csv(OUT_DIR / "training_curves.csv", curve_seed_rows + curve_agg_rows, curve_fields + [field for field in agg_fields if field not in curve_fields])
    _write_csv(OUT_DIR / "training_curves_aggregate.csv", curve_agg_rows, agg_fields)

    stage_rows = _stage_rows(schedules)
    seed_stage_fields = [
        "row_type",
        "model",
        "training_seed",
        "split",
        "progressive_stage",
        "stage_epoch_count",
        "stage_epoch_min",
        "stage_epoch_max",
        "stage_progress_min",
        "stage_progress_max",
        "effective_shield_mode",
        "effective_lookahead_horizon",
        "effective_risk_threshold",
        *TARGET_METRICS,
    ]
    agg_stage_fields = [
        "row_type",
        "model",
        "training_seed",
        "split",
        "progressive_stage",
        "stage_epoch_count_mean",
        "stage_epoch_count_std",
        "stage_epoch_min_mean",
        "stage_epoch_max_mean",
        "stage_progress_min_mean",
        "stage_progress_max_mean",
        "effective_shield_mode",
        "effective_lookahead_horizon_mean",
        "effective_lookahead_horizon_std",
        "effective_risk_threshold_mean",
        "effective_risk_threshold_std",
    ] + [field for metric in TARGET_METRICS for field in (f"{metric}_mean", f"{metric}_std")]
    _write_csv(OUT_DIR / "stage_metrics.csv", stage_rows, seed_stage_fields + [field for field in agg_stage_fields if field not in seed_stage_fields])

    summary_rows = _summary_rows(schedules)
    summary_fields = [
        "model",
        "training_seed_count",
        "eval_seed_count_per_checkpoint",
        "default_runtime_mode",
        "default_runtime_horizon",
        "default_runtime_threshold",
        "progressive_enabled",
        "progressive_early_mode",
        "progressive_early_end_ratio",
        "progressive_late_start_ratio",
        "progressive_early_risk_threshold",
        "progressive_mid_risk_threshold",
        "progressive_late_risk_threshold",
        "progressive_mid_lookahead_horizon",
        "progressive_late_lookahead_horizon",
        *FORMAL_METRICS,
    ]
    dynamic_summary_fields = sorted(
        {
            key
            for row in summary_rows
            for key in row.keys()
            if key not in summary_fields
        }
    )
    _write_csv(OUT_DIR / "summary_metrics.csv", summary_rows, summary_fields + dynamic_summary_fields)
    _write_simple_html(summary_rows, stage_rows)
    print(f"[progressive-mechanism] out={OUT_DIR}")


if __name__ == "__main__":
    main()
