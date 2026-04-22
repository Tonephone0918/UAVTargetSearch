from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np


# The *_safe_* fields below are kept as backward-compatible aliases for the
# always-on A_hard statistics that older CSV / plotting scripts may still read.
SUMMARY_CSV_FIELDS = [
    "split",
    "epoch",
    "phase",
    "mode",
    "hard_solver_mode",
    "recursive_gate_mode",
    "dead_end_policy",
    "risk_variant",
    "exact_diagnostics_enabled",
    "progressive_enabled",
    "episodes",
    "episode_steps",
    "total_steps",
    "episode_return",
    "original_episode_return",
    "avg_reward",
    "avg_original_reward",
    "search_rate",
    "found_targets",
    "coverage_ratio",
    "coverage_rate",
    "error_rate",
    "collision_count",
    "collisions",
    "shield_trigger_count",
    "shield_trigger_rate",
    "action_replacement_count",
    "action_replacement_rate",
    "episode_min_uav_margin",
    "episode_mean_uav_margin",
    "episode_min_threat_margin",
    "episode_mean_threat_margin",
    "near_miss_count",
    "near_miss_rate",
    "avg_hard_action_count",
    "avg_safe_action_count",
    "min_hard_action_count",
    "avg_rec_action_count",
    "min_safe_action_count",
    "min_rec_action_count",
    "dead_end_count",
    "dead_end_rate",
    "dead_end_hard_count",
    "dead_end_hard_rate",
    "dead_end_safe_count",
    "dead_end_safe_rate",
    "dead_end_rec_count",
    "dead_end_rec_rate",
    "exact_hard_query_count",
    "exact_hard_feasible_count",
    "exact_hard_rescue_count",
    "exact_hard_false_empty_count",
    "exact_hard_empty_query_count",
    "exact_hard_false_empty_rate",
    "exact_hard_action_count",
    "seq_exact_compare_count",
    "seq_empty_count",
    "seq_nonempty_count",
    "seq_empty_exact_nonempty_count",
    "seq_empty_exact_nonempty_rate",
    "seq_nonempty_exact_empty_count",
    "seq_nonempty_exact_empty_rate",
    "seq_exact_intersection_size",
    "seq_exact_jaccard",
    "hard_repair_attempt_count",
    "hard_repair_success_count",
    "hard_repair_success_rate",
    "emergency_count",
    "emergency_rate",
    "emergency_agent_count",
    "emergency_agent_rate",
    "guarantee_broken_count",
    "guarantee_broken_rate",
    "guarantee_broken_agent_count",
    "guarantee_broken_agent_rate",
    "shield_penalty_sum",
    "shield_penalty_rate",
    "shield_fallback_count",
    "shield_agent_trigger_count",
    "avg_risk_score",
    "avg_risk_clear",
    "avg_risk_clear_gap",
    "avg_risk_fragility",
    "avg_risk_region",
    "avg_risk_hist",
    "avg_risk_support",
    "high_risk_agent_count",
    "high_risk_rate",
    "recursive_gate_agent_count",
    "recursive_gate_rate",
    "future_witness_branch_count",
    "avg_future_witness_branch_count",
    "future_beam_width_used",
    "perf_step_time_ms",
    "perf_shield_time_ms",
    "perf_hard_time_ms",
    "perf_safe_time_ms",
    "perf_exact_hard_time_ms",
    "perf_rule_mask_time_ms",
    "perf_refine_time_ms",
    "perf_predict_time_ms",
    "perf_recursive_time_ms",
    "perf_stats_time_ms",
    "perf_steps_per_sec",
    "perf_shield_time_ratio",
    "perf_predict_cache_hit_rate",
    "perf_hard_cache_hit_rate",
    "perf_safe_cache_hit_rate",
    "perf_future_cache_hit_rate",
    "perf_recursive_gate_run_rate",
    "perf_recursive_gate_skip_rate",
    "perf_recursive_candidate_checks",
]


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.mean(values))


def _safe_min(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.min(values))


class EpisodeStatsAccumulator:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total_steps = 0
        self.episode_return = 0.0
        self.original_episode_return = 0.0
        self.collision_count = 0.0
        self.shield_trigger_count = 0
        self.action_replacement_count = 0
        self.shield_agent_trigger_count = 0
        self.shield_fallback_count = 0
        self.shield_penalty_sum = 0.0
        self.near_miss_count = 0
        self.dead_end_count = 0
        self.dead_end_hard_count = 0
        self.dead_end_safe_count = 0
        self.dead_end_rec_count = 0
        self.exact_hard_query_count = 0
        self.exact_hard_feasible_count = 0
        self.exact_hard_rescue_count = 0
        self.exact_hard_false_empty_count = 0
        self.exact_hard_empty_query_count = 0
        self.exact_hard_action_total = 0.0
        self.seq_exact_compare_count = 0
        self.seq_empty_count = 0
        self.seq_nonempty_count = 0
        self.seq_empty_exact_nonempty_count = 0
        self.seq_nonempty_exact_empty_count = 0
        self.seq_exact_intersection_size_total = 0.0
        self.seq_exact_jaccard_total = 0.0
        self.hard_repair_attempt_count = 0
        self.hard_repair_success_count = 0
        self.emergency_count = 0
        self.emergency_agent_count = 0
        self.guarantee_broken_count = 0
        self.guarantee_broken_agent_count = 0
        self.total_agent_steps = 0
        self.min_uav_margins: list[float] = []
        self.mean_uav_margins: list[float] = []
        self.min_threat_margins: list[float] = []
        self.mean_threat_margins: list[float] = []
        self.hard_action_counts: list[float] = []
        self.safe_action_counts: list[float] = []
        self.rec_action_counts: list[float] = []
        self.min_hard_action_counts: list[float] = []
        self.min_safe_action_counts: list[float] = []
        self.min_rec_action_counts: list[float] = []
        self.risk_scores: list[float] = []
        self.risk_clear_scores: list[float] = []
        self.risk_clear_gap_scores: list[float] = []
        self.risk_fragility_scores: list[float] = []
        self.risk_region_scores: list[float] = []
        self.risk_hist_scores: list[float] = []
        self.risk_support_scores: list[float] = []
        self.high_risk_agent_count = 0
        self.recursive_gate_agent_count = 0
        self.future_witness_branch_count = 0.0
        self.future_beam_width_used_values: list[float] = []

    def update(self, info: Dict[str, float]) -> None:
        self.total_steps += 1
        penalized_reward = float(info.get("shield_penalized_reward", info.get("reward", 0.0)))
        original_reward = float(info.get("original_reward", penalized_reward))
        self.episode_return += penalized_reward
        self.original_episode_return += original_reward
        self.collision_count += float(info.get("collisions", 0.0))
        self.shield_trigger_count += int(bool(info.get("shield_triggered", False)))
        self.action_replacement_count += int(bool(info.get("action_replaced", False)))
        self.shield_agent_trigger_count += int(info.get("shield_triggered_agents", 0))
        self.shield_fallback_count += int(bool(info.get("shield_fallback_triggered", False)))
        self.shield_penalty_sum += float(info.get("shield_penalty", 0.0))
        self.near_miss_count += int(bool(info.get("near_miss", False)))
        self.dead_end_count += int(bool(info.get("dead_end", False)))
        self.dead_end_hard_count += int(bool(info.get("dead_end_hard", info.get("dead_end_safe", False))))
        self.dead_end_safe_count += int(bool(info.get("dead_end_safe", False)))
        self.dead_end_rec_count += int(bool(info.get("dead_end_rec", False)))
        self.exact_hard_query_count += int(info.get("exact_hard_query_count_step", 0))
        self.exact_hard_feasible_count += int(info.get("exact_hard_feasible_count_step", 0))
        self.exact_hard_rescue_count += int(info.get("exact_hard_rescue_count_step", 0))
        self.exact_hard_false_empty_count += int(info.get("exact_hard_false_empty_count_step", 0))
        self.exact_hard_empty_query_count += int(info.get("exact_hard_empty_query_count_step", 0))
        self.exact_hard_action_total += float(info.get("exact_hard_action_count_step", 0.0))
        self.seq_exact_compare_count += int(info.get("seq_exact_compare_count_step", 0))
        self.seq_empty_count += int(info.get("seq_empty_count_step", 0))
        self.seq_nonempty_count += int(info.get("seq_nonempty_count_step", 0))
        self.seq_empty_exact_nonempty_count += int(info.get("seq_empty_exact_nonempty_count_step", 0))
        self.seq_nonempty_exact_empty_count += int(info.get("seq_nonempty_exact_empty_count_step", 0))
        self.seq_exact_intersection_size_total += float(info.get("seq_exact_intersection_size_step", 0.0))
        self.seq_exact_jaccard_total += float(info.get("seq_exact_jaccard_total_step", 0.0))
        self.hard_repair_attempt_count += int(info.get("hard_repair_attempt_count_step", 0))
        self.hard_repair_success_count += int(info.get("hard_repair_success_count_step", 0))
        self.emergency_count += int(bool(info.get("emergency_triggered", False)))
        self.emergency_agent_count += int(info.get("emergency_agents", 0))
        self.guarantee_broken_count += int(bool(info.get("guarantee_broken", False)))
        self.guarantee_broken_agent_count += int(info.get("guarantee_broken_agents", 0))
        self.min_uav_margins.append(float(info.get("min_uav_uav_margin", 0.0)))
        self.mean_uav_margins.append(float(info.get("mean_uav_uav_margin", info.get("min_uav_uav_margin", 0.0))))
        self.min_threat_margins.append(float(info.get("min_uav_threat_margin", 0.0)))
        self.mean_threat_margins.append(float(info.get("mean_uav_threat_margin", info.get("min_uav_threat_margin", 0.0))))
        self.hard_action_counts.append(float(info.get("hard_action_count", info.get("safe_action_count", 0.0))))
        self.safe_action_counts.append(float(info.get("safe_action_count", info.get("hard_action_count", 0.0))))
        self.rec_action_counts.append(float(info.get("rec_action_count", 0.0)))
        self.min_hard_action_counts.append(
            float(info.get("min_hard_action_count_step", info.get("min_safe_action_count_step", info.get("hard_action_count", info.get("safe_action_count", 0.0)))))
        )
        self.min_safe_action_counts.append(
            float(info.get("min_safe_action_count_step", info.get("safe_action_count", info.get("hard_action_count", 0.0))))
        )
        self.min_rec_action_counts.append(float(info.get("min_rec_action_count_step", info.get("rec_action_count", 0.0))))
        self.risk_scores.append(float(info.get("risk_score", 0.0)))
        self.risk_clear_scores.append(float(info.get("risk_clear", 0.0)))
        self.risk_clear_gap_scores.append(float(info.get("risk_clear_gap", 0.0)))
        self.risk_fragility_scores.append(float(info.get("risk_fragility", 0.0)))
        self.risk_region_scores.append(float(info.get("risk_region", 0.0)))
        self.risk_hist_scores.append(float(info.get("risk_hist", 0.0)))
        self.risk_support_scores.append(float(info.get("risk_support", 0.0)))
        self.high_risk_agent_count += int(info.get("high_risk_agents", 0))
        self.recursive_gate_agent_count += int(info.get("recursive_gate_agents", 0))
        self.future_witness_branch_count += float(info.get("future_witness_branch_count_step", 0.0))
        self.future_beam_width_used_values.append(float(info.get("future_beam_width_used_step", 0.0)))
        self.total_agent_steps += int(info.get("risk_agent_count", 0))

    def finalize(self, last_info: Dict[str, float], *, found_targets: int, shield_mode: str) -> Dict[str, float]:
        steps = max(1, self.total_steps)
        agent_steps = max(1, self.total_agent_steps)
        coverage_ratio = float(last_info.get("coverage_ratio", last_info.get("coverage_rate", 0.0)))
        metrics = {
            "episode_steps": float(self.total_steps),
            "total_steps": float(self.total_steps),
            "episode_return": float(self.episode_return),
            "original_episode_return": float(self.original_episode_return),
            "avg_reward": float(self.episode_return),
            "avg_original_reward": float(self.original_episode_return),
            "search_rate": float(last_info.get("search_rate", 0.0)),
            "found_targets": float(found_targets),
            "coverage_ratio": coverage_ratio,
            "coverage_rate": coverage_ratio,
            "error_rate": float(last_info.get("error_rate", 0.0)),
            "collision_count": float(self.collision_count),
            "collisions": float(self.collision_count),
            "shield_trigger_count": float(self.shield_trigger_count),
            "shield_trigger_rate": float(self.shield_trigger_count / steps),
            "action_replacement_count": float(self.action_replacement_count),
            "action_replacement_rate": float(self.action_replacement_count / steps),
            "episode_min_uav_margin": _safe_min(self.min_uav_margins),
            "episode_mean_uav_margin": _safe_mean(self.mean_uav_margins),
            "episode_min_threat_margin": _safe_min(self.min_threat_margins),
            "episode_mean_threat_margin": _safe_mean(self.mean_threat_margins),
            "near_miss_count": float(self.near_miss_count),
            "near_miss_rate": float(self.near_miss_count / steps),
            "avg_hard_action_count": _safe_mean(self.hard_action_counts),
            "avg_safe_action_count": _safe_mean(self.safe_action_counts),
            "avg_rec_action_count": _safe_mean(self.rec_action_counts),
            "min_hard_action_count": _safe_min(self.min_hard_action_counts),
            "min_safe_action_count": _safe_min(self.min_safe_action_counts),
            "min_rec_action_count": _safe_min(self.min_rec_action_counts),
            "dead_end_count": float(self.dead_end_count),
            "dead_end_rate": float(self.dead_end_count / steps),
            "dead_end_hard_count": float(self.dead_end_hard_count),
            "dead_end_hard_rate": float(self.dead_end_hard_count / steps),
            "dead_end_safe_count": float(self.dead_end_safe_count),
            "dead_end_safe_rate": float(self.dead_end_safe_count / steps),
            "dead_end_rec_count": float(self.dead_end_rec_count),
            "dead_end_rec_rate": float(self.dead_end_rec_count / steps),
            "exact_hard_query_count": float(self.exact_hard_query_count),
            "exact_hard_feasible_count": float(self.exact_hard_feasible_count),
            "exact_hard_rescue_count": float(self.exact_hard_rescue_count),
            "exact_hard_false_empty_count": float(self.exact_hard_false_empty_count),
            "exact_hard_empty_query_count": float(self.exact_hard_empty_query_count),
            "exact_hard_false_empty_rate": float(
                self.exact_hard_false_empty_count / max(self.exact_hard_empty_query_count, 1)
            ),
            "exact_hard_action_count": float(self.exact_hard_action_total / max(self.exact_hard_query_count, 1)),
            "seq_exact_compare_count": float(self.seq_exact_compare_count),
            "seq_empty_count": float(self.seq_empty_count),
            "seq_nonempty_count": float(self.seq_nonempty_count),
            "seq_empty_exact_nonempty_count": float(self.seq_empty_exact_nonempty_count),
            "seq_empty_exact_nonempty_rate": float(
                self.seq_empty_exact_nonempty_count / max(self.seq_empty_count, 1)
            ),
            "seq_nonempty_exact_empty_count": float(self.seq_nonempty_exact_empty_count),
            "seq_nonempty_exact_empty_rate": float(
                self.seq_nonempty_exact_empty_count / max(self.seq_nonempty_count, 1)
            ),
            # Mean intersection size per compared agent-step.
            "seq_exact_intersection_size": float(
                self.seq_exact_intersection_size_total / max(self.seq_exact_compare_count, 1)
            ),
            # Mean Jaccard over compared agent-steps; both-empty is defined as 1.
            "seq_exact_jaccard": float(self.seq_exact_jaccard_total / max(self.seq_exact_compare_count, 1)),
            "hard_repair_attempt_count": float(self.hard_repair_attempt_count),
            "hard_repair_success_count": float(self.hard_repair_success_count),
            "hard_repair_success_rate": float(self.hard_repair_success_count / max(self.hard_repair_attempt_count, 1)),
            "emergency_count": float(self.emergency_count),
            "emergency_rate": float(self.emergency_count / steps),
            "emergency_agent_count": float(self.emergency_agent_count),
            "emergency_agent_rate": float(self.emergency_agent_count / agent_steps),
            "guarantee_broken_count": float(self.guarantee_broken_count),
            "guarantee_broken_rate": float(self.guarantee_broken_count / steps),
            "guarantee_broken_agent_count": float(self.guarantee_broken_agent_count),
            "guarantee_broken_agent_rate": float(self.guarantee_broken_agent_count / agent_steps),
            "shield_penalty_sum": float(self.shield_penalty_sum),
            "shield_penalty_rate": float(self.shield_penalty_sum / steps),
            "shield_fallback_count": float(self.shield_fallback_count),
            "shield_agent_trigger_count": float(self.shield_agent_trigger_count),
            "avg_risk_score": _safe_mean(self.risk_scores),
            "avg_risk_clear": _safe_mean(self.risk_clear_scores),
            "avg_risk_clear_gap": _safe_mean(self.risk_clear_gap_scores),
            "avg_risk_fragility": _safe_mean(self.risk_fragility_scores),
            "avg_risk_region": _safe_mean(self.risk_region_scores),
            "avg_risk_hist": _safe_mean(self.risk_hist_scores),
            "avg_risk_support": _safe_mean(self.risk_support_scores),
            "high_risk_agent_count": float(self.high_risk_agent_count),
            "high_risk_rate": float(self.high_risk_agent_count / agent_steps),
            "recursive_gate_agent_count": float(self.recursive_gate_agent_count),
            "recursive_gate_rate": float(self.recursive_gate_agent_count / agent_steps),
            "future_witness_branch_count": float(self.future_witness_branch_count),
            "avg_future_witness_branch_count": float(self.future_witness_branch_count / steps),
            "future_beam_width_used": _safe_mean(self.future_beam_width_used_values),
            "mode_is_recursive": float(shield_mode == "recursive"),
        }
        return metrics


class SummaryCSVLogger:
    def __init__(self, log_dir: str):
        self.path = Path(log_dir) / "metrics_summary.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, object]) -> None:
        exists = self.path.exists()
        payload = {field: row.get(field, "") for field in SUMMARY_CSV_FIELDS}
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(payload)


def log_summary_scalars(writer, split: str, metrics: Dict[str, float], step: int) -> None:
    scalar_keys: Iterable[str] = [
        "original_episode_return",
        "found_targets",
        "coverage_ratio",
        "error_rate",
        "collision_count",
        "shield_trigger_rate",
        "action_replacement_rate",
        "episode_min_uav_margin",
        "episode_mean_uav_margin",
        "episode_min_threat_margin",
        "episode_mean_threat_margin",
        "near_miss_rate",
        "avg_hard_action_count",
        "avg_safe_action_count",
        "avg_rec_action_count",
        "min_hard_action_count",
        "min_safe_action_count",
        "min_rec_action_count",
        "dead_end_rate",
        "dead_end_hard_rate",
        "dead_end_safe_rate",
        "dead_end_rec_rate",
        "exact_hard_query_count",
        "exact_hard_feasible_count",
        "exact_hard_rescue_count",
        "exact_hard_false_empty_count",
        "exact_hard_empty_query_count",
        "exact_hard_false_empty_rate",
        "exact_hard_action_count",
        "seq_exact_compare_count",
        "seq_empty_count",
        "seq_nonempty_count",
        "seq_empty_exact_nonempty_count",
        "seq_empty_exact_nonempty_rate",
        "seq_nonempty_exact_empty_count",
        "seq_nonempty_exact_empty_rate",
        "seq_exact_intersection_size",
        "seq_exact_jaccard",
        "hard_repair_attempt_count",
        "hard_repair_success_count",
        "hard_repair_success_rate",
        "emergency_rate",
        "emergency_agent_rate",
        "guarantee_broken_rate",
        "guarantee_broken_agent_rate",
        "shield_penalty_sum",
        "shield_penalty_rate",
        "shield_fallback_count",
        "avg_risk_score",
        "avg_risk_clear",
        "avg_risk_clear_gap",
        "avg_risk_fragility",
        "avg_risk_region",
        "avg_risk_hist",
        "avg_risk_support",
        "high_risk_agent_count",
        "high_risk_rate",
        "recursive_gate_agent_count",
        "recursive_gate_rate",
        "future_witness_branch_count",
        "avg_future_witness_branch_count",
        "future_beam_width_used",
        "total_steps",
        "episode_steps",
        "perf_step_time_ms",
        "perf_shield_time_ms",
        "perf_hard_time_ms",
        "perf_safe_time_ms",
        "perf_exact_hard_time_ms",
        "perf_rule_mask_time_ms",
        "perf_refine_time_ms",
        "perf_predict_time_ms",
        "perf_recursive_time_ms",
        "perf_stats_time_ms",
        "perf_steps_per_sec",
        "perf_shield_time_ratio",
        "perf_predict_cache_hit_rate",
        "perf_hard_cache_hit_rate",
        "perf_safe_cache_hit_rate",
        "perf_future_cache_hit_rate",
        "perf_recursive_gate_run_rate",
        "perf_recursive_gate_skip_rate",
        "perf_recursive_candidate_checks",
    ]
    for key in scalar_keys:
        if key in metrics:
            writer.add_scalar(f"{split}/{key}", float(metrics[key]), step)

    # Backward-compatible aliases for the existing plots.
    writer.add_scalar(f"{split}/coverage_rate", float(metrics["coverage_rate"]), step)
    writer.add_scalar(f"{split}/collisions", float(metrics["collisions"]), step)
    if split == "eval":
        writer.add_scalar("eval/avg_reward", float(metrics["avg_reward"]), step)
    if split == "train":
        writer.add_scalar("shield/fallback_count", float(metrics["shield_fallback_count"]), step)
        writer.add_scalar("shield/emergency_rate", float(metrics.get("emergency_rate", 0.0)), step)
        writer.add_scalar("shield/guarantee_broken_rate", float(metrics.get("guarantee_broken_rate", 0.0)), step)
        writer.add_scalar("shield/penalty", float(metrics["shield_penalty_sum"]), step)
        writer.add_scalar("shield/a_hard_size", float(metrics["avg_hard_action_count"]), step)
        writer.add_scalar("shield/a_safe_size", float(metrics["avg_safe_action_count"]), step)
        writer.add_scalar("shield/a_rec_size", float(metrics["avg_rec_action_count"]), step)
        writer.add_scalar("shield/risk_score", float(metrics["avg_risk_score"]), step)
        writer.add_scalar("shield/risk_clear", float(metrics["avg_risk_clear"]), step)
        writer.add_scalar("shield/risk_clear_gap", float(metrics.get("avg_risk_clear_gap", 0.0)), step)
        writer.add_scalar("shield/risk_fragility", float(metrics.get("avg_risk_fragility", 0.0)), step)
        writer.add_scalar("shield/risk_support", float(metrics.get("avg_risk_support", 0.0)), step)
        writer.add_scalar("shield/high_risk_rate", float(metrics["high_risk_rate"]), step)
        writer.add_scalar("shield/recursive_gate_rate", float(metrics["recursive_gate_rate"]), step)
        writer.add_scalar("shield/min_uav_uav_margin", float(metrics["episode_min_uav_margin"]), step)
        writer.add_scalar("shield/min_uav_threat_margin", float(metrics["episode_min_threat_margin"]), step)
