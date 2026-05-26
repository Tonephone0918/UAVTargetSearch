from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        return None
    return float(text)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary_index(path: Path) -> dict[str, dict[str, float | str | None]]:
    rows = _load_csv(path)
    out: dict[str, dict[str, float | str | None]] = {}
    for row in rows:
        model = row["model"]
        parsed: dict[str, float | str | None] = {}
        for key, value in row.items():
            if key in {"model", "checkpoint"}:
                parsed[key] = value
            else:
                parsed[key] = _to_float(value)
        out[model] = parsed
    return out


def _delta_per_ms(delta_metric: float, delta_ms: float) -> float | None:
    if abs(delta_ms) <= 1e-9:
        return None
    return float(delta_metric / delta_ms)


def _tradeoff_row(
    *,
    family: str,
    baseline_name: str,
    candidate_name: str,
    baseline: dict[str, float | str | None],
    candidate: dict[str, float | str | None],
) -> dict[str, object]:
    baseline_collision = float(baseline.get("collision_count_mean", 0.0) or 0.0)
    baseline_guarantee = float(baseline.get("guarantee_broken_rate_mean", 0.0) or 0.0)
    baseline_dead_end_rec = float(baseline.get("dead_end_rec_rate_mean", 0.0) or 0.0)
    baseline_search = float(baseline.get("search_rate_mean", 0.0) or 0.0)
    baseline_recursive_ms = float(baseline.get("perf_recursive_time_ms_mean", 0.0) or 0.0)
    baseline_shield_ms = float(baseline.get("perf_shield_time_ms_mean", 0.0) or 0.0)

    candidate_collision = float(candidate.get("collision_count_mean", 0.0) or 0.0)
    candidate_guarantee = float(candidate.get("guarantee_broken_rate_mean", 0.0) or 0.0)
    candidate_dead_end_rec = float(candidate.get("dead_end_rec_rate_mean", 0.0) or 0.0)
    candidate_search = float(candidate.get("search_rate_mean", 0.0) or 0.0)
    candidate_recursive_ms = float(candidate.get("perf_recursive_time_ms_mean", 0.0) or 0.0)
    candidate_shield_ms = float(candidate.get("perf_shield_time_ms_mean", 0.0) or 0.0)

    delta_collision = candidate_collision - baseline_collision
    delta_guarantee = candidate_guarantee - baseline_guarantee
    delta_dead_end_rec = candidate_dead_end_rec - baseline_dead_end_rec
    delta_search = candidate_search - baseline_search
    delta_recursive_ms = candidate_recursive_ms - baseline_recursive_ms
    delta_shield_ms = candidate_shield_ms - baseline_shield_ms

    return {
        "family": family,
        "baseline_model": baseline_name,
        "candidate_model": candidate_name,
        "baseline_collision_count": baseline_collision,
        "candidate_collision_count": candidate_collision,
        "baseline_guarantee_broken_rate": baseline_guarantee,
        "candidate_guarantee_broken_rate": candidate_guarantee,
        "baseline_dead_end_rec_rate": baseline_dead_end_rec,
        "candidate_dead_end_rec_rate": candidate_dead_end_rec,
        "baseline_search_rate": baseline_search,
        "candidate_search_rate": candidate_search,
        "baseline_perf_recursive_time_ms": baseline_recursive_ms,
        "candidate_perf_recursive_time_ms": candidate_recursive_ms,
        "baseline_perf_shield_time_ms": baseline_shield_ms,
        "candidate_perf_shield_time_ms": candidate_shield_ms,
        "delta_collision": delta_collision,
        "delta_guarantee_broken_rate": delta_guarantee,
        "delta_dead_end_rec_rate": delta_dead_end_rec,
        "delta_search_rate": delta_search,
        "delta_perf_recursive_time_ms": delta_recursive_ms,
        "delta_perf_shield_time_ms": delta_shield_ms,
        "delta_collision_per_shield_ms": _delta_per_ms(delta_collision, delta_shield_ms),
        "delta_guarantee_broken_rate_per_shield_ms": _delta_per_ms(delta_guarantee, delta_shield_ms),
        "delta_collision_per_recursive_ms": _delta_per_ms(delta_collision, delta_recursive_ms),
        "delta_guarantee_broken_rate_per_recursive_ms": _delta_per_ms(delta_guarantee, delta_recursive_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute simple unit-time H=2 tradeoff metrics from validate-only summary CSVs.")
    parser.add_argument(
        "--compare-summary",
        type=str,
        required=True,
        help="Summary CSV from compare_lookahead_horizons.py after retiming.",
    )
    parser.add_argument(
        "--risk-scan-summary",
        type=str,
        required=True,
        help="Summary CSV from scan_h2_risk_thresholds.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="runs/h2_tradeoff_analysis/tradeoff_summary.csv",
        help="Output CSV for derived tradeoff metrics.",
    )
    args = parser.parse_args()

    compare = _build_summary_index(Path(args.compare_summary))
    risk_scan = _build_summary_index(Path(args.risk_scan_summary))

    rows: list[dict[str, object]] = []
    if "recursive_full_rescue_h1" in compare and "recursive_full_rescue_h2" in compare:
        rows.append(
            _tradeoff_row(
                family="recursive_full",
                baseline_name="recursive_full_rescue_h1",
                candidate_name="recursive_full_rescue_h2",
                baseline=compare["recursive_full_rescue_h1"],
                candidate=compare["recursive_full_rescue_h2"],
            )
        )

    if "recursive_risk_rescue_h1" in compare and "recursive_risk_rescue_h2" in compare:
        rows.append(
            _tradeoff_row(
                family="recursive_risk_default_eta035",
                baseline_name="recursive_risk_rescue_h1",
                candidate_name="recursive_risk_rescue_h2",
                baseline=compare["recursive_risk_rescue_h1"],
                candidate=compare["recursive_risk_rescue_h2"],
            )
        )

    risk_baseline = risk_scan.get("recursive_risk_rescue_h1_eta035")
    if risk_baseline is not None:
        for model_name, row in sorted(risk_scan.items()):
            if not model_name.startswith("recursive_risk_rescue_h2_eta"):
                continue
            rows.append(
                _tradeoff_row(
                    family="recursive_risk_h2_threshold_scan",
                    baseline_name="recursive_risk_rescue_h1_eta035",
                    candidate_name=model_name,
                    baseline=risk_baseline,
                    candidate=row,
                )
            )

    fieldnames = [
        "family",
        "baseline_model",
        "candidate_model",
        "baseline_collision_count",
        "candidate_collision_count",
        "baseline_guarantee_broken_rate",
        "candidate_guarantee_broken_rate",
        "baseline_dead_end_rec_rate",
        "candidate_dead_end_rec_rate",
        "baseline_search_rate",
        "candidate_search_rate",
        "baseline_perf_recursive_time_ms",
        "candidate_perf_recursive_time_ms",
        "baseline_perf_shield_time_ms",
        "candidate_perf_shield_time_ms",
        "delta_collision",
        "delta_guarantee_broken_rate",
        "delta_dead_end_rec_rate",
        "delta_search_rate",
        "delta_perf_recursive_time_ms",
        "delta_perf_shield_time_ms",
        "delta_collision_per_shield_ms",
        "delta_guarantee_broken_rate_per_shield_ms",
        "delta_collision_per_recursive_ms",
        "delta_guarantee_broken_rate_per_recursive_ms",
    ]
    output_csv = Path(args.output_csv)
    _write_csv(output_csv, rows, fieldnames)
    print(f"[h2-tradeoff] output={output_csv}")
    for row in rows:
        print(
            "[h2-tradeoff] "
            + ", ".join(
                [
                    f"family={row['family']}",
                    f"candidate={row['candidate_model']}",
                    f"delta_collision={float(row['delta_collision']):.6f}",
                    f"delta_guarantee={float(row['delta_guarantee_broken_rate']):.6f}",
                    f"delta_recursive_ms={float(row['delta_perf_recursive_time_ms']):.6f}",
                    f"delta_shield_ms={float(row['delta_perf_shield_time_ms']):.6f}",
                    f"delta_collision_per_shield_ms={row['delta_collision_per_shield_ms']}",
                ]
            )
        )


if __name__ == "__main__":
    main()
