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
DEFAULT_THRESHOLDS = [0.35, 0.45, 0.55, 0.65]


def _load_render_module(root: Path):
    module_path = root / "Playground" / "render_model_compare_page.py"
    spec = importlib.util.spec_from_file_location("render_model_compare_page", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _metric_delta(candidate: float, baseline: float) -> float:
    return float(candidate - baseline)


def _benefit_per_ms(improvement: float, extra_ms: float) -> float | None:
    if extra_ms <= 1e-9:
        return None
    return float(improvement / extra_ms)


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


def _add_tradeoff_columns(summary_rows: list[dict[str, object]], *, baseline_model: str) -> None:
    baseline = next((row for row in summary_rows if row["model"] == baseline_model), None)
    if baseline is None:
        return

    baseline_collision = float(baseline.get("collision_count_mean", 0.0))
    baseline_guarantee = float(baseline.get("guarantee_broken_rate_mean", 0.0))
    baseline_dead_end_rec = float(baseline.get("dead_end_rec_rate_mean", 0.0))
    baseline_gate = float(baseline.get("recursive_gate_rate_mean", 0.0))
    baseline_avg_rec = float(baseline.get("avg_rec_action_count_mean", 0.0))
    baseline_recursive_ms = float(baseline.get("perf_recursive_time_ms_mean", 0.0))
    baseline_shield_ms = float(baseline.get("perf_shield_time_ms_mean", 0.0))
    baseline_search = float(baseline.get("search_rate_mean", 0.0))

    for row in summary_rows:
        candidate_collision = float(row.get("collision_count_mean", 0.0))
        candidate_guarantee = float(row.get("guarantee_broken_rate_mean", 0.0))
        candidate_dead_end_rec = float(row.get("dead_end_rec_rate_mean", 0.0))
        candidate_gate = float(row.get("recursive_gate_rate_mean", 0.0))
        candidate_avg_rec = float(row.get("avg_rec_action_count_mean", 0.0))
        candidate_recursive_ms = float(row.get("perf_recursive_time_ms_mean", 0.0))
        candidate_shield_ms = float(row.get("perf_shield_time_ms_mean", 0.0))
        candidate_search = float(row.get("search_rate_mean", 0.0))

        delta_collision = _metric_delta(candidate_collision, baseline_collision)
        delta_guarantee = _metric_delta(candidate_guarantee, baseline_guarantee)
        delta_dead_end_rec = _metric_delta(candidate_dead_end_rec, baseline_dead_end_rec)
        delta_gate = _metric_delta(candidate_gate, baseline_gate)
        delta_avg_rec = _metric_delta(candidate_avg_rec, baseline_avg_rec)
        delta_recursive_ms = _metric_delta(candidate_recursive_ms, baseline_recursive_ms)
        delta_shield_ms = _metric_delta(candidate_shield_ms, baseline_shield_ms)
        delta_search = _metric_delta(candidate_search, baseline_search)

        collision_improvement = -delta_collision
        guarantee_improvement = -delta_guarantee

        row["delta_collision_vs_h1"] = delta_collision
        row["delta_guarantee_broken_rate_vs_h1"] = delta_guarantee
        row["delta_dead_end_rec_rate_vs_h1"] = delta_dead_end_rec
        row["delta_recursive_gate_rate_vs_h1"] = delta_gate
        row["delta_avg_rec_action_count_vs_h1"] = delta_avg_rec
        row["delta_perf_recursive_time_ms_vs_h1"] = delta_recursive_ms
        row["delta_perf_shield_time_ms_vs_h1"] = delta_shield_ms
        row["delta_search_rate_vs_h1"] = delta_search
        row["collision_reduction_per_shield_ms"] = _benefit_per_ms(collision_improvement, delta_shield_ms)
        row["guarantee_reduction_per_shield_ms"] = _benefit_per_ms(guarantee_improvement, delta_shield_ms)
        row["collision_reduction_per_recursive_ms"] = _benefit_per_ms(collision_improvement, delta_recursive_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate-only H=2 risk-threshold scan on a fixed recursive(risk)+rescue checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Checkpoint to evaluate.")
    parser.add_argument("--episodes", type=int, default=2, help="Evaluation episodes per seed.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2], help="Explicit seed list.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=DEFAULT_THRESHOLDS,
        help="Risk thresholds to scan for H=2.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/h2_risk_threshold_scan",
        help="Directory for per-seed and summary CSV outputs.",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="runs/vis/h2_risk_threshold_scan.html",
        help="Rendered comparison HTML page.",
    )
    args = parser.parse_args()

    checkpoint_path = str((ROOT / args.checkpoint).resolve()) if not Path(args.checkpoint).is_absolute() else args.checkpoint
    render_module = _load_render_module(ROOT)
    per_seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    shared_overrides = {
        "enabled": True,
        "mode": "recursive",
        "profile_enabled": True,
        "dead_end_policy": "fail_closed",
        "hard_solver_mode": "sequential_with_exact_rescue",
        "risk_variant": "risk_base",
        "recursive_gate_mode": "risk",
    }

    model_specs: list[tuple[str, dict[str, object]]] = [
        (
            "recursive_risk_rescue_h1_eta035",
            {
                **shared_overrides,
                "lookahead_horizon": 1,
                "risk_threshold": 0.35,
            },
        )
    ]
    for threshold in args.thresholds:
        tag = str(int(round(float(threshold) * 100.0))).zfill(2)
        model_specs.append(
            (
                f"recursive_risk_rescue_h2_eta{tag}",
                {
                    **shared_overrides,
                    "lookahead_horizon": 2,
                    "risk_threshold": float(threshold),
                },
            )
        )

    for model_name, shield_overrides in model_specs:
        print(f"[h2-risk-scan] start model={model_name}")
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
                "[h2-risk-scan] "
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

    _add_tradeoff_columns(summary_rows, baseline_model="recursive_risk_rescue_h1_eta035")

    output_dir = ROOT / args.output_dir
    per_seed_csv = output_dir / "per_seed_metrics.csv"
    summary_csv = output_dir / "summary_metrics.csv"
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
    _write_csv(per_seed_csv, per_seed_rows, per_seed_fields)
    _write_csv(summary_csv, summary_rows, summary_fields)

    output_html = ROOT / args.output_html
    render_module.build_report(
        summary_csv=summary_csv,
        per_seed_csv=per_seed_csv,
        output_html=output_html,
        compare_return=False,
        title="H=2 Risk-Threshold Scan",
        subtitle=(
            f"validate-only, checkpoint={Path(checkpoint_path).name}, "
            f"seeds={list(args.seeds)}, episodes={int(args.episodes)}"
        ),
    )
    print(f"[h2-risk-scan] per-seed={per_seed_csv}")
    print(f"[h2-risk-scan] summary={summary_csv}")
    print(f"[h2-risk-scan] html={output_html}")


if __name__ == "__main__":
    main()
