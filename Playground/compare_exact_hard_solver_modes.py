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


DEFAULT_CHECKPOINTS = {
    "safe": "checkpoints/baseline_mappo_safe_normdpm_dense2000/best.pt",
    "recursive_full": "checkpoints/baseline_mappo_recursive_full_riskbase_normdpm_dense2000/best.pt",
    "recursive_risk": "checkpoints/baseline_mappo_recursive_risk_riskbase_normdpm_dense2000/best.pt",
}


def _load_render_module(root: Path):
    module_path = root / "Playground" / "render_model_compare_page.py"
    spec = importlib.util.spec_from_file_location("render_model_compare_page", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _model_specs(root: Path) -> list[tuple[str, str, dict[str, object]]]:
    shared = {
        "enabled": True,
        "profile_enabled": True,
        "dead_end_policy": "fail_closed",
        "exact_diagnostics_enabled": True,
    }
    return [
        (
            "safe_sequential",
            str(root / DEFAULT_CHECKPOINTS["safe"]),
            {
                **shared,
                "mode": "safe",
                "hard_solver_mode": "sequential",
            },
        ),
        (
            "safe_exact",
            str(root / DEFAULT_CHECKPOINTS["safe"]),
            {
                **shared,
                "mode": "safe",
                "hard_solver_mode": "exact",
            },
        ),
        (
            "safe_rescue",
            str(root / DEFAULT_CHECKPOINTS["safe"]),
            {
                **shared,
                "mode": "safe",
                "hard_solver_mode": "sequential_with_exact_rescue",
            },
        ),
        (
            "recursive_full_sequential",
            str(root / DEFAULT_CHECKPOINTS["recursive_full"]),
            {
                **shared,
                "mode": "recursive",
                "recursive_gate_mode": "full",
                "risk_variant": "risk_base",
                "hard_solver_mode": "sequential",
            },
        ),
        (
            "recursive_full_exact",
            str(root / DEFAULT_CHECKPOINTS["recursive_full"]),
            {
                **shared,
                "mode": "recursive",
                "recursive_gate_mode": "full",
                "risk_variant": "risk_base",
                "hard_solver_mode": "exact",
            },
        ),
        (
            "recursive_full_rescue",
            str(root / DEFAULT_CHECKPOINTS["recursive_full"]),
            {
                **shared,
                "mode": "recursive",
                "recursive_gate_mode": "full",
                "risk_variant": "risk_base",
                "hard_solver_mode": "sequential_with_exact_rescue",
            },
        ),
        (
            "recursive_risk_sequential",
            str(root / DEFAULT_CHECKPOINTS["recursive_risk"]),
            {
                **shared,
                "mode": "recursive",
                "recursive_gate_mode": "risk",
                "risk_variant": "risk_base",
                "hard_solver_mode": "sequential",
            },
        ),
        (
            "recursive_risk_exact",
            str(root / DEFAULT_CHECKPOINTS["recursive_risk"]),
            {
                **shared,
                "mode": "recursive",
                "recursive_gate_mode": "risk",
                "risk_variant": "risk_base",
                "hard_solver_mode": "exact",
            },
        ),
        (
            "recursive_risk_rescue",
            str(root / DEFAULT_CHECKPOINTS["recursive_risk"]),
            {
                **shared,
                "mode": "recursive",
                "recursive_gate_mode": "risk",
                "risk_variant": "risk_base",
                "hard_solver_mode": "sequential_with_exact_rescue",
            },
        ),
    ]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate-only compare for sequential vs exact vs rescue A_hard solver modes.")
    parser.add_argument("--episodes", type=int, default=2, help="Evaluation episodes per seed.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2], help="Explicit seed list.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/model_compare_exact_hard_solver_diag_medium2x2",
        help="Directory for summary/per-seed CSV outputs.",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="runs/vis/model_compare_exact_hard_solver_diag_medium2x2.html",
        help="Rendered comparison HTML page.",
    )
    args = parser.parse_args()

    root = ROOT
    render_module = _load_render_module(root)

    per_seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    model_specs = _model_specs(root)

    for model_name, checkpoint_path, shield_overrides in model_specs:
        print(f"[compare] start model={model_name} checkpoint={checkpoint_path}")
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
            }
            row.update({key: float(value) for key, value in metrics.items()})
            per_seed_rows.append(row)
            print(
                "[compare] "
                + ", ".join(
                    [
                        f"model={model_name}",
                        f"seed={seed}",
                        f"collision_count={metrics.get('collision_count', 0.0):.4f}",
                        f"dead_end_hard_rate={metrics.get('dead_end_hard_rate', 0.0):.4f}",
                        f"guarantee_broken_rate={metrics.get('guarantee_broken_rate', 0.0):.4f}",
                        f"exact_hard_false_empty_rate={metrics.get('exact_hard_false_empty_rate', 0.0):.4f}",
                        f"seq_nonempty_exact_empty_rate={metrics.get('seq_nonempty_exact_empty_rate', 0.0):.4f}",
                        f"seq_exact_jaccard={metrics.get('seq_exact_jaccard', 0.0):.4f}",
                        f"perf_exact_hard_time_ms={metrics.get('perf_exact_hard_time_ms', 0.0):.4f}",
                    ]
                )
            )
        summary_row: dict[str, object] = {
            "model": model_name,
            "checkpoint": checkpoint_path,
            "num_seeds": len(args.seeds),
            "episodes_per_seed": int(args.episodes),
        }
        metric_names = sorted(rows[0].keys()) if rows else []
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in rows]
            summary_row[f"{metric_name}_mean"] = mean(values)
            summary_row[f"{metric_name}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary_row)

    output_dir = root / args.output_dir
    per_seed_csv = output_dir / "per_seed_metrics.csv"
    summary_csv = output_dir / "summary_metrics.csv"
    per_seed_fields = ["model", "checkpoint", "seed", "episodes"] + sorted(
        {
            key
            for row in per_seed_rows
            for key in row.keys()
            if key not in {"model", "checkpoint", "seed", "episodes"}
        }
    )
    summary_fields = ["model", "checkpoint", "num_seeds", "episodes_per_seed"] + sorted(
        {
            key
            for row in summary_rows
            for key in row.keys()
            if key not in {"model", "checkpoint", "num_seeds", "episodes_per_seed"}
        }
    )
    _write_csv(per_seed_csv, per_seed_rows, per_seed_fields)
    _write_csv(summary_csv, summary_rows, summary_fields)

    output_html = root / args.output_html
    render_module.build_report(
        summary_csv=summary_csv,
        per_seed_csv=per_seed_csv,
        output_html=output_html,
        compare_return=False,
        title="Exact A_hard Solver Compare",
        subtitle=f"validate-only, seeds={list(args.seeds)}, episodes={int(args.episodes)}, diagnostics=on",
    )
    print(f"[compare] per-seed={per_seed_csv}")
    print(f"[compare] summary={summary_csv}")
    print(f"[compare] html={output_html}")


if __name__ == "__main__":
    main()
