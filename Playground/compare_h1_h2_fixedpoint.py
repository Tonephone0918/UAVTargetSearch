from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hrvdn.validate import evaluate_checkpoint


DEFAULT_CHECKPOINT = "checkpoints/baseline_mappo_recursive_risk_rescue_riskbase_normdpm_dense2000/best.pt"
DEFAULT_H2_THRESHOLD = 0.55
DEFAULT_H1_THRESHOLDS = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]


def _load_render_module(root: Path):
    module_path = root / "Playground" / "render_model_compare_page.py"
    spec = importlib.util.spec_from_file_location("render_model_compare_page", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(
    *,
    model: str,
    checkpoint: str,
    lookahead_horizon: int,
    risk_threshold: float,
    episodes: int,
    seeds: list[int],
    rows: list[dict[str, float]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "model": model,
        "checkpoint": checkpoint,
        "num_seeds": len(seeds),
        "episodes_per_seed": episodes,
        "lookahead_horizon": int(lookahead_horizon),
        "risk_threshold": float(risk_threshold),
    }
    metric_names = sorted(rows[0].keys()) if rows else []
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows]
        summary[f"{metric_name}_mean"] = mean(values)
        summary[f"{metric_name}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return summary


def _specs(h2_threshold: float, h1_thresholds: list[float]) -> list[tuple[str, dict[str, object]]]:
    shared = {
        "enabled": True,
        "mode": "recursive",
        "profile_enabled": True,
        "dead_end_policy": "fail_closed",
        "hard_solver_mode": "sequential_with_exact_rescue",
        "risk_variant": "risk_base",
        "recursive_gate_mode": "risk",
    }
    specs: list[tuple[str, dict[str, object]]] = [
        (
            f"recursive_risk_rescue_h2_eta{int(round(h2_threshold * 100.0)):02d}_ref",
            {
                **shared,
                "lookahead_horizon": 2,
                "risk_threshold": float(h2_threshold),
            },
        )
    ]
    for threshold in h1_thresholds:
        specs.append(
            (
                f"recursive_risk_rescue_h1_eta{int(round(threshold * 100.0)):02d}",
                {
                    **shared,
                    "lookahead_horizon": 1,
                    "risk_threshold": float(threshold),
                },
            )
        )
    return specs


def _metric(row: dict[str, object], key: str) -> float:
    return float(row.get(key, 0.0))


def _match_rows(
    *,
    summary_rows: list[dict[str, object]],
    reference_model: str,
    match_key: str,
    output_key: str,
    top_k: int = 2,
) -> list[dict[str, object]]:
    reference = next(row for row in summary_rows if row["model"] == reference_model)
    candidates = [row for row in summary_rows if row["model"] != reference_model]
    ref_value = _metric(reference, match_key)
    ordered = sorted(candidates, key=lambda row: abs(_metric(row, match_key) - ref_value))
    selected = ordered[: max(1, int(top_k))]

    result_rows: list[dict[str, object]] = []
    for row in selected:
        candidate_value = _metric(row, match_key)
        result_rows.append(
            {
                "reference_model": reference_model,
                "candidate_model": row["model"],
                "match_key": match_key,
                output_key: abs(candidate_value - ref_value),
                "reference_risk_threshold": reference["risk_threshold"],
                "candidate_risk_threshold": row["risk_threshold"],
                "reference_perf_shield_time_ms": _metric(reference, "perf_shield_time_ms_mean"),
                "candidate_perf_shield_time_ms": _metric(row, "perf_shield_time_ms_mean"),
                "reference_recursive_gate_rate": _metric(reference, "recursive_gate_rate_mean"),
                "candidate_recursive_gate_rate": _metric(row, "recursive_gate_rate_mean"),
                "reference_collision_count": _metric(reference, "collision_count_mean"),
                "candidate_collision_count": _metric(row, "collision_count_mean"),
                "reference_guarantee_broken_rate": _metric(reference, "guarantee_broken_rate_mean"),
                "candidate_guarantee_broken_rate": _metric(row, "guarantee_broken_rate_mean"),
                "reference_dead_end_rec_rate": _metric(reference, "dead_end_rec_rate_mean"),
                "candidate_dead_end_rec_rate": _metric(row, "dead_end_rec_rate_mean"),
                "reference_avg_rec_action_count": _metric(reference, "avg_rec_action_count_mean"),
                "candidate_avg_rec_action_count": _metric(row, "avg_rec_action_count_mean"),
                "reference_search_rate": _metric(reference, "search_rate_mean"),
                "candidate_search_rate": _metric(row, "search_rate_mean"),
                "delta_collision_h2_minus_h1": _metric(reference, "collision_count_mean") - _metric(row, "collision_count_mean"),
                "delta_guarantee_h2_minus_h1": _metric(reference, "guarantee_broken_rate_mean") - _metric(row, "guarantee_broken_rate_mean"),
                "delta_dead_end_rec_h2_minus_h1": _metric(reference, "dead_end_rec_rate_mean") - _metric(row, "dead_end_rec_rate_mean"),
                "delta_perf_shield_time_h2_minus_h1": _metric(reference, "perf_shield_time_ms_mean") - _metric(row, "perf_shield_time_ms_mean"),
                "delta_recursive_gate_rate_h2_minus_h1": _metric(reference, "recursive_gate_rate_mean") - _metric(row, "recursive_gate_rate_mean"),
                "delta_search_rate_h2_minus_h1": _metric(reference, "search_rate_mean") - _metric(row, "search_rate_mean"),
            }
        )
    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair validate-only comparison: fixed H2 reference point vs H1 eta scan.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Checkpoint to evaluate.")
    parser.add_argument("--episodes", type=int, default=3, help="Evaluation episodes per seed.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3], help="Explicit seed list.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--h2-threshold", type=float, default=DEFAULT_H2_THRESHOLD, help="Fixed H2 reference eta.")
    parser.add_argument(
        "--h1-thresholds",
        type=float,
        nargs="*",
        default=DEFAULT_H1_THRESHOLDS,
        help="H1 eta scan values.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/h1_h2_fixedpoint_compare",
        help="Directory for per-seed/summary/match CSV outputs.",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="runs/vis/h1_h2_fixedpoint_compare.html",
        help="Rendered comparison HTML page.",
    )
    args = parser.parse_args()

    checkpoint_path = str((ROOT / args.checkpoint).resolve()) if not Path(args.checkpoint).is_absolute() else args.checkpoint
    render_module = _load_render_module(ROOT)
    per_seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    specs = _specs(float(args.h2_threshold), [float(v) for v in args.h1_thresholds])
    for model_name, shield_overrides in specs:
        print(f"[h1-h2-fixed] start model={model_name}")
        rows: list[dict[str, float]] = []
        for seed in args.seeds:
            metrics = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                episodes=int(args.episodes),
                device=args.device,
                env_overrides={"seed": int(seed)},
                shield_overrides=shield_overrides,
            )
            rows.append(metrics)
            row: dict[str, object] = {
                "model": model_name,
                "checkpoint": checkpoint_path,
                "seed": int(seed),
                "episodes": int(args.episodes),
                "lookahead_horizon": int(shield_overrides["lookahead_horizon"]),
                "risk_threshold": float(shield_overrides["risk_threshold"]),
            }
            row.update({key: float(value) for key, value in metrics.items()})
            per_seed_rows.append(row)
            print(
                "[h1-h2-fixed] "
                + ", ".join(
                    [
                        f"model={model_name}",
                        f"seed={seed}",
                        f"collision_count={metrics.get('collision_count', 0.0):.4f}",
                        f"guarantee_broken_rate={metrics.get('guarantee_broken_rate', 0.0):.4f}",
                        f"dead_end_rec_rate={metrics.get('dead_end_rec_rate', 0.0):.4f}",
                        f"recursive_gate_rate={metrics.get('recursive_gate_rate', 0.0):.4f}",
                        f"avg_rec_action_count={metrics.get('avg_rec_action_count', 0.0):.4f}",
                        f"perf_recursive_time_ms={metrics.get('perf_recursive_time_ms', 0.0):.4f}",
                        f"perf_recursive_work_time_ms={metrics.get('perf_recursive_work_time_ms', 0.0):.4f}",
                        f"perf_shield_time_ms={metrics.get('perf_shield_time_ms', 0.0):.4f}",
                        f"search_rate={metrics.get('search_rate', 0.0):.4f}",
                    ]
                )
            )
        summary_rows.append(
            _summary_row(
                model=model_name,
                checkpoint=checkpoint_path,
                lookahead_horizon=int(shield_overrides["lookahead_horizon"]),
                risk_threshold=float(shield_overrides["risk_threshold"]),
                episodes=int(args.episodes),
                seeds=[int(seed) for seed in args.seeds],
                rows=rows,
            )
        )

    h2_reference_model = f"recursive_risk_rescue_h2_eta{int(round(float(args.h2_threshold) * 100.0)):02d}_ref"
    matched_compute_rows = _match_rows(
        summary_rows=summary_rows,
        reference_model=h2_reference_model,
        match_key="perf_shield_time_ms_mean",
        output_key="abs_perf_shield_time_gap",
        top_k=2,
    )
    matched_gate_rows = _match_rows(
        summary_rows=summary_rows,
        reference_model=h2_reference_model,
        match_key="recursive_gate_rate_mean",
        output_key="abs_recursive_gate_rate_gap",
        top_k=2,
    )

    output_dir = ROOT / args.output_dir
    per_seed_csv = output_dir / "per_seed_metrics.csv"
    summary_csv = output_dir / "summary_metrics.csv"
    matched_compute_csv = output_dir / "matched_compute_budget.csv"
    matched_gate_csv = output_dir / "matched_gate_rate.csv"

    per_seed_fields = [
        "model",
        "checkpoint",
        "seed",
        "episodes",
        "lookahead_horizon",
        "risk_threshold",
    ] + sorted(
        {
            key
            for row in per_seed_rows
            for key in row.keys()
            if key not in {"model", "checkpoint", "seed", "episodes", "lookahead_horizon", "risk_threshold"}
        }
    )
    summary_fields = [
        "model",
        "checkpoint",
        "num_seeds",
        "episodes_per_seed",
        "lookahead_horizon",
        "risk_threshold",
    ] + sorted(
        {
            key
            for row in summary_rows
            for key in row.keys()
            if key not in {"model", "checkpoint", "num_seeds", "episodes_per_seed", "lookahead_horizon", "risk_threshold"}
        }
    )
    match_fields = list(matched_compute_rows[0].keys()) if matched_compute_rows else []

    _write_csv(per_seed_csv, per_seed_rows, per_seed_fields)
    _write_csv(summary_csv, summary_rows, summary_fields)
    if matched_compute_rows:
        _write_csv(matched_compute_csv, matched_compute_rows, match_fields)
    if matched_gate_rows:
        _write_csv(matched_gate_csv, matched_gate_rows, list(matched_gate_rows[0].keys()))

    output_html = ROOT / args.output_html
    render_module.build_report(
        summary_csv=summary_csv,
        per_seed_csv=per_seed_csv,
        output_html=output_html,
        compare_return=False,
        title="Fixed H2 Point vs H1 Eta Scan",
        subtitle=(
            f"validate-only, checkpoint={Path(checkpoint_path).name}, "
            f"seeds={list(args.seeds)}, episodes={int(args.episodes)}, "
            f"H2 eta={float(args.h2_threshold):.2f}"
        ),
    )

    print(f"[h1-h2-fixed] per-seed={per_seed_csv}")
    print(f"[h1-h2-fixed] summary={summary_csv}")
    print(f"[h1-h2-fixed] matched-compute={matched_compute_csv}")
    print(f"[h1-h2-fixed] matched-gate={matched_gate_csv}")
    print(f"[h1-h2-fixed] html={output_html}")


if __name__ == "__main__":
    main()
