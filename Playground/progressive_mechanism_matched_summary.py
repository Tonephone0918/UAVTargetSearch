from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "progressive_mechanism_20260428"
PILOT_PARTIALS = OUT_DIR / "matched_pilot_nonprog_threshold_scan_3x3_partials"
FORMAL_ETA25_PARTIALS = OUT_DIR / "matched_formal_nonprog_eta25_5x5_partials"

MATCHED_KEYS = [
    "collision_count_mean",
    "search_rate_mean",
    "guarantee_broken_rate_mean",
    "dead_end_rec_rate_mean",
    "recursive_gate_rate_mean",
    "perf_shield_time_ms_mean",
    "perf_recursive_time_ms_mean",
]


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


def _combine_single_model_partials(partials_dir: Path, out_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    per_seed_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    for path in sorted(partials_dir.glob("*/per_seed_metrics.csv")):
        per_seed_rows.extend(_read_csv(path))
    for path in sorted(partials_dir.glob("*/summary_metrics.csv")):
        summary_rows.extend(_read_csv(path))

    if per_seed_rows:
        per_seed_fields = ["model", "checkpoint", "seed", "episodes"] + sorted(
            {k for row in per_seed_rows for k in row.keys()} - {"model", "checkpoint", "seed", "episodes"}
        )
        _write_csv(out_dir / "per_seed_metrics.csv", per_seed_rows, per_seed_fields)
    if summary_rows:
        summary_fields = ["model", "checkpoint", "num_seeds", "episodes_per_seed"] + sorted(
            {k for row in summary_rows for k in row.keys()} - {"model", "checkpoint", "num_seeds", "episodes_per_seed"}
        )
        _write_csv(out_dir / "summary_metrics.csv", summary_rows, summary_fields)
    return per_seed_rows, summary_rows


def _aggregate_training_seed_summaries(summary_rows: list[dict[str, str]], model_name: str) -> dict[str, object]:
    out: dict[str, object] = {
        "model": model_name,
        "training_seed_count": len(summary_rows),
    }
    if not summary_rows:
        return out
    out["eval_seed_count_per_checkpoint"] = int(summary_rows[0]["num_seeds"])
    out["episodes_per_seed"] = int(summary_rows[0]["episodes_per_seed"])
    for key in MATCHED_KEYS:
        vals = [_to_float(row.get(key)) for row in summary_rows]
        vals = [v for v in vals if v is not None]
        out[key] = mean(vals) if vals else 0.0
        out[key.replace("_mean", "_std")] = pstdev(vals) if len(vals) > 1 else 0.0
    return out


def main() -> None:
    pilot_dir = OUT_DIR / "matched_pilot_nonprog_threshold_scan_3x3"
    pilot_per_seed, pilot_summary = _combine_single_model_partials(PILOT_PARTIALS, pilot_dir)

    formal_eta25_dir = OUT_DIR / "matched_formal_nonprog_eta25_5x5"
    formal_eta25_per_seed, formal_eta25_summary = _combine_single_model_partials(FORMAL_ETA25_PARTIALS, formal_eta25_dir)

    base_summary_rows = {row["model"]: row for row in _read_csv(OUT_DIR / "summary_metrics.csv")}
    eta25_aggregate = _aggregate_training_seed_summaries(formal_eta25_summary, "non_progressive_eta25")

    matched_rows: list[dict[str, object]] = []
    for row in sorted(pilot_summary, key=lambda r: r["model"]):
        out: dict[str, object] = {
            "analysis_type": "pilot_scan",
            "budget": "3_eval_seeds_x_3_episodes",
            "reference_model": "threshold_only_progressive_seed1_fixed_h1_eta035",
            "candidate_model": row["model"],
            "candidate_threshold": row["model"].replace("nonprog_eta", "0.").replace("threshold_only_ref", "0.35")
            if row["model"] != "threshold_only_ref"
            else "0.35",
        }
        for key in MATCHED_KEYS:
            out[key] = _to_float(row.get(key))
        matched_rows.append(out)

    threshold_ref = base_summary_rows["threshold_only_progressive"]
    nonprog_ref = base_summary_rows["non_progressive"]
    gate_row = {
        "analysis_type": "formal_matched_gate_rate",
        "budget": "3_training_seeds_x_5_eval_seeds_x_5_episodes",
        "reference_model": "threshold_only_progressive",
        "candidate_model": "non_progressive_default_eta035",
        "candidate_threshold": 0.35,
        "reference_recursive_gate_rate_mean": _to_float(threshold_ref["recursive_gate_rate_mean"]),
        "candidate_recursive_gate_rate_mean": _to_float(nonprog_ref["recursive_gate_rate_mean"]),
        "abs_recursive_gate_rate_gap": abs(
            float(threshold_ref["recursive_gate_rate_mean"]) - float(nonprog_ref["recursive_gate_rate_mean"])
        ),
        "reference_collision_count_mean": _to_float(threshold_ref["collision_count_mean"]),
        "candidate_collision_count_mean": _to_float(nonprog_ref["collision_count_mean"]),
        "reference_search_rate_mean": _to_float(threshold_ref["search_rate_mean"]),
        "candidate_search_rate_mean": _to_float(nonprog_ref["search_rate_mean"]),
        "reference_guarantee_broken_rate_mean": _to_float(threshold_ref["guarantee_broken_rate_mean"]),
        "candidate_guarantee_broken_rate_mean": _to_float(nonprog_ref["guarantee_broken_rate_mean"]),
        "reference_dead_end_rec_rate_mean": _to_float(threshold_ref["dead_end_rec_rate_mean"]),
        "candidate_dead_end_rec_rate_mean": _to_float(nonprog_ref["dead_end_rec_rate_mean"]),
        "reference_perf_shield_time_ms_mean": _to_float(threshold_ref["perf_shield_time_ms_mean"]),
        "candidate_perf_shield_time_ms_mean": _to_float(nonprog_ref["perf_shield_time_ms_mean"]),
        "reference_perf_recursive_time_ms_mean": _to_float(threshold_ref["perf_recursive_time_ms_mean"]),
        "candidate_perf_recursive_time_ms_mean": _to_float(nonprog_ref["perf_recursive_time_ms_mean"]),
        "delta_collision_threshold_minus_candidate": float(threshold_ref["collision_count_mean"]) - float(nonprog_ref["collision_count_mean"]),
        "delta_guarantee_threshold_minus_candidate": float(threshold_ref["guarantee_broken_rate_mean"]) - float(nonprog_ref["guarantee_broken_rate_mean"]),
        "delta_dead_end_rec_threshold_minus_candidate": float(threshold_ref["dead_end_rec_rate_mean"]) - float(nonprog_ref["dead_end_rec_rate_mean"]),
        "delta_perf_shield_time_threshold_minus_candidate": float(threshold_ref["perf_shield_time_ms_mean"]) - float(nonprog_ref["perf_shield_time_ms_mean"]),
    }
    matched_rows.append(gate_row)

    compute_row = {
        "analysis_type": "formal_matched_compute_budget",
        "budget": "3_training_seeds_x_5_eval_seeds_x_5_episodes",
        "reference_model": "threshold_only_progressive",
        "candidate_model": "non_progressive_eta025",
        "candidate_threshold": 0.25,
        "reference_recursive_gate_rate_mean": _to_float(threshold_ref["recursive_gate_rate_mean"]),
        "candidate_recursive_gate_rate_mean": eta25_aggregate["recursive_gate_rate_mean"],
        "abs_recursive_gate_rate_gap": abs(
            float(threshold_ref["recursive_gate_rate_mean"]) - float(eta25_aggregate["recursive_gate_rate_mean"])
        ),
        "reference_collision_count_mean": _to_float(threshold_ref["collision_count_mean"]),
        "candidate_collision_count_mean": eta25_aggregate["collision_count_mean"],
        "reference_search_rate_mean": _to_float(threshold_ref["search_rate_mean"]),
        "candidate_search_rate_mean": eta25_aggregate["search_rate_mean"],
        "reference_guarantee_broken_rate_mean": _to_float(threshold_ref["guarantee_broken_rate_mean"]),
        "candidate_guarantee_broken_rate_mean": eta25_aggregate["guarantee_broken_rate_mean"],
        "reference_dead_end_rec_rate_mean": _to_float(threshold_ref["dead_end_rec_rate_mean"]),
        "candidate_dead_end_rec_rate_mean": eta25_aggregate["dead_end_rec_rate_mean"],
        "reference_perf_shield_time_ms_mean": _to_float(threshold_ref["perf_shield_time_ms_mean"]),
        "candidate_perf_shield_time_ms_mean": eta25_aggregate["perf_shield_time_ms_mean"],
        "reference_perf_recursive_time_ms_mean": _to_float(threshold_ref["perf_recursive_time_ms_mean"]),
        "candidate_perf_recursive_time_ms_mean": eta25_aggregate["perf_recursive_time_ms_mean"],
        "abs_perf_shield_time_gap": abs(
            float(threshold_ref["perf_shield_time_ms_mean"]) - float(eta25_aggregate["perf_shield_time_ms_mean"])
        ),
        "delta_collision_threshold_minus_candidate": float(threshold_ref["collision_count_mean"]) - float(eta25_aggregate["collision_count_mean"]),
        "delta_guarantee_threshold_minus_candidate": float(threshold_ref["guarantee_broken_rate_mean"]) - float(eta25_aggregate["guarantee_broken_rate_mean"]),
        "delta_dead_end_rec_threshold_minus_candidate": float(threshold_ref["dead_end_rec_rate_mean"]) - float(eta25_aggregate["dead_end_rec_rate_mean"]),
        "delta_perf_shield_time_threshold_minus_candidate": float(threshold_ref["perf_shield_time_ms_mean"]) - float(eta25_aggregate["perf_shield_time_ms_mean"]),
    }
    matched_rows.append(compute_row)

    fields = ["analysis_type", "budget", "reference_model", "candidate_model", "candidate_threshold"] + sorted(
        {key for row in matched_rows for key in row.keys()} - {"analysis_type", "budget", "reference_model", "candidate_model", "candidate_threshold"}
    )
    _write_csv(OUT_DIR / "matched_analysis_summary.csv", matched_rows, fields)
    print(f"[matched-summary] out={OUT_DIR / 'matched_analysis_summary.csv'}")


if __name__ == "__main__":
    main()
