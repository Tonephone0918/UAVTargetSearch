from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections import Counter
from numbers import Real
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

from .validate import evaluate_checkpoint


def _parse_checkpoint_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Checkpoint argument must be NAME=PATH, got: {raw}")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Checkpoint argument must be NAME=PATH, got: {raw}")
    return name, path


def _parse_override_value(raw: str):
    value = raw.strip()
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_shield_override_arg(raw: str) -> tuple[str, Dict[str, object]]:
    if "=" not in raw:
        raise ValueError(f"Shield override must be NAME=key=value[,key=value...], got: {raw}")
    name, spec = raw.split("=", 1)
    name = name.strip()
    if not name or not spec.strip():
        raise ValueError(f"Shield override must be NAME=key=value[,key=value...], got: {raw}")
    overrides: Dict[str, object] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Shield override entry must be key=value, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Shield override key cannot be empty in: {raw}")
        overrides[key] = _parse_override_value(value)
    if not overrides:
        raise ValueError(f"Shield override must include at least one key=value pair, got: {raw}")
    return name, overrides


def _parse_seeds(args) -> List[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    start = int(args.seed_start)
    count = int(args.num_seeds)
    return list(range(start, start + count))


def _metric_keys(rows: Iterable[Dict[str, float]]) -> List[str]:
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return sorted(keys)


def _is_numeric_metric(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_multiseed_eval(
    checkpoints: List[Tuple[str, str]],
    seeds: List[int],
    episodes: int,
    device: str,
    shield_overrides_by_model: Dict[str, Dict[str, object]] | None = None,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    per_seed_rows: List[Dict[str, object]] = []
    grouped_metrics: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for model_name, checkpoint_path in checkpoints:
        shield_overrides = (shield_overrides_by_model or {}).get(model_name)
        for seed in seeds:
            metrics = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                episodes=episodes,
                device=device,
                env_overrides={"seed": seed},
                shield_overrides=shield_overrides,
            )
            grouped_metrics[model_name].append(metrics)
            row: Dict[str, object] = {
                "model": model_name,
                "checkpoint": checkpoint_path,
                "seed": seed,
                "episodes": episodes,
            }
            row.update(
                {
                    key: (float(value) if _is_numeric_metric(value) else value)
                    for key, value in metrics.items()
                }
            )
            per_seed_rows.append(row)

    summary_rows: List[Dict[str, object]] = []
    for model_name, rows in grouped_metrics.items():
        metric_names = _metric_keys(rows)
        summary: Dict[str, object] = {
            "model": model_name,
            "checkpoint": next(path for name, path in checkpoints if name == model_name),
            "num_seeds": len(rows),
            "episodes_per_seed": episodes,
        }
        for metric in metric_names:
            values = [row[metric] for row in rows if metric in row]
            if values and all(_is_numeric_metric(value) for value in values):
                numeric_values = [float(value) for value in values]
                summary[f"{metric}_mean"] = mean(numeric_values)
                summary[f"{metric}_std"] = pstdev(numeric_values) if len(numeric_values) > 1 else 0.0
            elif values:
                summary[metric] = Counter(str(value) for value in values).most_common(1)[0][0]
        summary_rows.append(summary)

    return per_seed_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one or more checkpoints across multiple seeds and aggregate the metrics.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint spec in the form NAME=PATH. Pass this flag multiple times.",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes per seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed-start", type=int, default=1, help="First seed when --seeds is not provided.")
    parser.add_argument("--num-seeds", type=int, default=20, help="Number of consecutive seeds when --seeds is not provided.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Explicit seed list. Overrides --seed-start/--num-seeds.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/multiseed_eval",
        help="Directory for per-seed and summary CSV outputs.",
    )
    parser.add_argument(
        "--shield-override",
        action="append",
        default=[],
        help="Per-model shield override in the form NAME=key=value[,key=value...]. Pass multiple times if needed.",
    )
    args = parser.parse_args()

    checkpoints = [_parse_checkpoint_arg(raw) for raw in args.checkpoint]
    checkpoint_names = {name for name, _ in checkpoints}
    shield_overrides_by_model: Dict[str, Dict[str, object]] = {}
    for raw in args.shield_override:
        name, overrides = _parse_shield_override_arg(raw)
        if name not in checkpoint_names:
            raise ValueError(f"Shield override references unknown model '{name}'. Known models: {sorted(checkpoint_names)}")
        shield_overrides_by_model.setdefault(name, {}).update(overrides)
    seeds = _parse_seeds(args)
    per_seed_rows, summary_rows = run_multiseed_eval(
        checkpoints=checkpoints,
        seeds=seeds,
        episodes=int(args.episodes),
        device=args.device,
        shield_overrides_by_model=shield_overrides_by_model,
    )

    output_dir = Path(args.output_dir)
    per_seed_fields = ["model", "checkpoint", "seed", "episodes"] + _metric_keys(
        [{key: value for key, value in row.items() if key not in {"model", "checkpoint", "seed", "episodes"}} for row in per_seed_rows]
    )
    summary_fields = ["model", "checkpoint", "num_seeds", "episodes_per_seed"] + _metric_keys(
        [{key: value for key, value in row.items() if key not in {"model", "checkpoint", "num_seeds", "episodes_per_seed"}} for row in summary_rows]
    )
    _write_csv(output_dir / "per_seed_metrics.csv", per_seed_rows, per_seed_fields)
    _write_csv(output_dir / "summary_metrics.csv", summary_rows, summary_fields)

    print(f"[multiseed-eval] seeds={seeds}")
    print(f"[multiseed-eval] per-seed={output_dir / 'per_seed_metrics.csv'}")
    print(f"[multiseed-eval] summary={output_dir / 'summary_metrics.csv'}")
    for model_name, overrides in shield_overrides_by_model.items():
        print(f"[multiseed-eval] shield_override[{model_name}]={overrides}")
    for row in summary_rows:
        display_keys = [
            "search_rate_mean",
            "coverage_ratio_mean",
            "collision_count_mean",
            "shield_trigger_rate_mean",
            "action_replacement_rate_mean",
            "near_miss_rate_mean",
            "dead_end_hard_rate_mean",
            "dead_end_safe_rate_mean",
            "dead_end_rec_rate_mean",
        ]
        parts = [f"model={row['model']}"]
        for key in display_keys:
            if key in row:
                parts.append(f"{key}={float(row[key]):.6f}")
        print("[multiseed-eval] " + ", ".join(parts))


if __name__ == "__main__":
    main()
