from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hrvdn.env import UAVSearchEnv
from src.hrvdn.runtime import (
    apply_env_overrides,
    build_mappo_from_env,
    build_policy_from_env,
    config_from_dict,
    load_checkpoint_module,
    load_checkpoint_policy,
    resolve_device,
)
from src.hrvdn.shield import CentralizedSafetyShield


@dataclass
class OfflineSafeRecord:
    episode: int
    step: int
    agent_idx: int
    proposed_action: int
    final_safe_action: int
    proposed_in_a_hard: int
    proposed_in_a_rec_oracle: int
    need_rec: int
    set_diff: int
    action_replaced: int
    fallback_triggered: int
    valid_action_count: int
    a_hard_size: int
    a_rec_oracle_size: int
    hard_set_fragility: float
    min_candidate_clearance: float
    max_candidate_clearance: float
    proposed_action_clearance: float
    proposed_action_clearance_available: int
    risk_clear_v1: float
    risk_clear_prop: float
    risk_clear_gap: float
    risk_region: float
    risk_hist: float
    risk_support: float
    near_boundary: int
    local_threat_count: int
    crowded: int


@dataclass(frozen=True)
class RiskVariant:
    name: str
    description: str
    weight_clear_min: float
    weight_clear_prop: float
    weight_clear_gap: float
    weight_fragility: float
    weight_region: float
    weight_hist: float
    weight_support: float


class PolicyRunner:
    def __init__(self, algo: str, module, env: UAVSearchEnv, device: str):
        self.algo = algo
        self.module = module
        self.env = env
        self.device = device
        self.hidden_states: list[torch.Tensor] | None = None
        self.reset_episode()

    def reset_episode(self) -> None:
        if self.algo == "hrvdn":
            hidden_size = self.module.gru.hidden_size
            self.hidden_states = [
                torch.zeros(1, 1, hidden_size, device=self.device) for _ in range(self.env.cfg.n_uavs)
            ]
        else:
            self.hidden_states = None

    @torch.no_grad()
    def propose(self, obs: Sequence[Dict[str, np.ndarray]]) -> tuple[List[int], torch.Tensor, np.ndarray]:
        if self.algo == "mappo":
            maps = torch.tensor(np.stack([o["map"] for o in obs]), dtype=torch.float32, device=self.device)
            extras = torch.tensor(np.stack([o["extra"] for o in obs]), dtype=torch.float32, device=self.device)
            masks = torch.tensor(np.stack([o["action_mask"] for o in obs]), dtype=torch.bool, device=self.device)
            logits = self.module(maps, extras)
            logits = logits.masked_fill(~masks, -1e9)
            acts = logits.argmax(dim=-1).tolist()
            return [int(a) for a in acts], logits, masks.detach().cpu().numpy()

        assert self.hidden_states is not None
        acts: List[int] = []
        preferences: List[torch.Tensor] = []
        valid_masks: List[np.ndarray] = []
        for i, o in enumerate(obs):
            om = torch.tensor(o["map"], dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
            ex = torch.tensor(o["extra"], dtype=torch.float32, device=self.device).unsqueeze(0)
            q, self.hidden_states[i] = self.module(om, ex, self.hidden_states[i])
            mask = torch.tensor(o["action_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
            q = q.masked_fill(~mask, -1e9)
            preferences.append(q.squeeze(0))
            valid_masks.append(np.asarray(o["action_mask"], dtype=bool))
            acts.append(int(q.argmax(dim=-1).item()))
        return acts, torch.stack(preferences, dim=0), np.stack(valid_masks)


def load_eval_stack(
    checkpoint_path: str,
    *,
    device: str,
    env_overrides: Dict[str, Any] | None = None,
) -> tuple[UAVSearchEnv, PolicyRunner, CentralizedSafetyShield]:
    run_device = resolve_device(device)
    ckpt = torch.load(checkpoint_path, map_location=run_device)
    algo = ckpt.get("algo", "hrvdn")
    cfg = config_from_dict(ckpt.get("config", {}))
    cfg.reward.mode = ckpt.get("reward_mode", cfg.reward.mode)
    apply_env_overrides(cfg, **(env_overrides or {}))

    cfg.shield.enabled = True
    cfg.shield.mode = "safe"
    cfg.shield.risk_score_enabled = True
    cfg.shield.legacy_recursive_gate = False

    env = UAVSearchEnv(cfg.env, cfg.reward, seed=cfg.train.seed)
    shield = CentralizedSafetyShield(cfg)
    if algo == "mappo":
        actor, critic = build_mappo_from_env(cfg, env, run_device)
        load_checkpoint_module(actor, ckpt["actor_state_dict"], checkpoint_path, "MAPPO actor")
        actor.eval()
        critic.eval()
        runner = PolicyRunner(algo, actor, env, run_device)
    else:
        policy = build_policy_from_env(cfg, env, run_device)
        load_checkpoint_policy(policy, ckpt["policy_state_dict"], checkpoint_path)
        policy.eval()
        runner = PolicyRunner(algo, policy, env, run_device)
    return env, runner, shield


def _sorted_unique_ints(values: Iterable[int]) -> List[int]:
    return sorted({int(v) for v in values})


def collect_safe_trajectory_records(
    checkpoint_path: str,
    *,
    episodes: int,
    device: str,
    env_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    env, runner, shield = load_eval_stack(checkpoint_path, device=device, env_overrides=env_overrides)
    records: List[OfflineSafeRecord] = []
    total_steps = 0
    wall_start = perf_counter()

    for episode_idx in range(episodes):
        obs = env.reset()
        runner.reset_episode()
        shield.reset_episode()
        done = False
        step_idx = 0
        while not done:
            proposed_actions, actor_preferences, action_masks = runner.propose(obs)
            proposed = [int(a) for a in proposed_actions]
            valid_masks = np.asarray(action_masks, dtype=bool)
            state = shield._capture_state(env)
            final_actions = list(proposed)
            planned_next_positions = shield._planned_next_positions(state, final_actions)

            shield_triggered = False
            triggered_agents = 0
            for agent_idx in range(shield.cfg.env.n_uavs):
                base_actions = list(final_actions)
                proposed_action = int(final_actions[agent_idx])
                hard_actions, hard_meta = shield._enumerate_hard_actions_with_meta(
                    env,
                    state,
                    agent_idx,
                    base_actions,
                    planned_next_positions=planned_next_positions,
                )
                oracle_rec_actions = shield.enumerate_recursive_feasible_actions(
                    env,
                    state,
                    agent_idx,
                    base_actions,
                    hard_actions,
                ) if hard_actions else []
                oracle_rec_actions = _sorted_unique_ints(oracle_rec_actions)

                proposed_in_a_hard = int(proposed_action in hard_actions)
                proposed_in_a_rec_oracle = int(proposed_action in oracle_rec_actions)
                need_rec = int(bool(proposed_in_a_hard and not proposed_in_a_rec_oracle))
                set_diff = int(_sorted_unique_ints(hard_actions) != oracle_rec_actions)

                risk_clear_v1 = float(shield.compute_clear_risk(hard_actions, hard_meta))
                risk_region = float(shield.compute_region_risk(hard_meta))
                risk_hist = float(shield.compute_hist_risk(agent_idx))
                risk_clear_prop = float(
                    shield.compute_proposed_clear_risk(
                        proposed_action,
                        hard_actions,
                        hard_meta,
                    )
                )
                risk_clear_gap = float(
                    shield.compute_clear_gap_risk(
                        proposed_action,
                        hard_actions,
                        hard_meta,
                    )
                )
                risk_support = float(shield.compute_support_risk(hard_actions, hard_meta))
                proposed_action_clearance = float(hard_meta.get("clearances", {}).get(proposed_action, float("nan")))
                proposed_action_clearance_available = int(proposed_action in hard_meta.get("clearances", {}))

                proposed_is_admissible = proposed_action in hard_actions if hard_actions else False
                selected_action = proposed_action
                action_replaced = 0
                fallback_triggered = 0
                if proposed_is_admissible:
                    shield.agent_intervention_history[agent_idx].append(0)
                else:
                    shield_triggered = True
                    triggered_agents += 1
                    if hard_actions:
                        selected_action, _ = shield.resample_action_from_allowed_set(
                            actor_preferences[agent_idx],
                            hard_actions,
                            valid_masks[agent_idx],
                            selection_mode="argmax",
                        )
                    else:
                        fallback_triggered = 1
                        shield.fallback_count += 1
                        fallback_candidates = np.flatnonzero(valid_masks[agent_idx]).tolist()
                        if fallback_candidates:
                            selected_action, _ = shield.resample_action_from_allowed_set(
                                actor_preferences[agent_idx],
                                fallback_candidates,
                                valid_masks[agent_idx],
                                selection_mode="argmax",
                            )
                        else:
                            selected_action = proposed_action
                    if int(selected_action) != proposed_action:
                        action_replaced = 1
                        shield.action_replaced_count += 1
                    final_actions[agent_idx] = int(selected_action)
                    planned_next_positions[agent_idx] = shield._single_next_position(state, agent_idx, int(selected_action))
                    shield.agent_intervention_history[agent_idx].append(action_replaced)

                records.append(
                    OfflineSafeRecord(
                        episode=int(episode_idx),
                        step=int(step_idx),
                        agent_idx=int(agent_idx),
                        proposed_action=int(proposed_action),
                        final_safe_action=int(selected_action),
                        proposed_in_a_hard=int(proposed_in_a_hard),
                        proposed_in_a_rec_oracle=int(proposed_in_a_rec_oracle),
                        need_rec=int(need_rec),
                        set_diff=int(set_diff),
                        action_replaced=int(action_replaced),
                        fallback_triggered=int(fallback_triggered),
                        valid_action_count=int(valid_masks[agent_idx].sum()),
                        a_hard_size=int(len(hard_actions)),
                        a_rec_oracle_size=int(len(oracle_rec_actions)),
                        hard_set_fragility=float(
                            1.0 - float(len(hard_actions)) / max(float(valid_masks[agent_idx].sum()), 1.0)
                        ),
                        min_candidate_clearance=float(hard_meta.get("min_candidate_clearance", float("nan"))),
                        max_candidate_clearance=float(hard_meta.get("max_candidate_clearance", float("nan"))),
                        proposed_action_clearance=float(proposed_action_clearance),
                        proposed_action_clearance_available=int(proposed_action_clearance_available),
                        risk_clear_v1=float(risk_clear_v1),
                        risk_clear_prop=float(risk_clear_prop),
                        risk_clear_gap=float(risk_clear_gap),
                        risk_region=float(risk_region),
                        risk_hist=float(risk_hist),
                        risk_support=float(risk_support),
                        near_boundary=int(bool(hard_meta.get("near_boundary", False))),
                        local_threat_count=int(hard_meta.get("local_threat_count", 0)),
                        crowded=int(bool(hard_meta.get("crowded", False))),
                    )
                )

            if shield_triggered:
                shield.shield_trigger_count += 1
                shield.shield_agent_trigger_count += triggered_agents
            shield.recent_trigger_history.append(1 if shield_triggered else 0)
            obs, _, done, _ = env.step(final_actions)
            total_steps += 1
            step_idx += 1

    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "episodes": int(episodes),
        "env_steps": int(total_steps),
        "wall_time_sec": float(perf_counter() - wall_start),
        "records": records,
        "shield_config": {
            "risk_variant": str(shield.cfg.shield.risk_variant),
            "risk_weight_clear": float(shield.cfg.shield.risk_weight_clear),
            "risk_weight_region": float(shield.cfg.shield.risk_weight_region),
            "risk_weight_hist": float(shield.cfg.shield.risk_weight_hist),
            "risk_clearance_norm": float(shield.cfg.shield.risk_clearance_norm),
            "risk_clear_gap_norm": float(shield.cfg.shield.risk_clear_gap_norm),
            "risk_support_clearance_margin": float(shield.cfg.shield.risk_support_clearance_margin),
            "risk_vnext_weight_prop_clear": float(shield.cfg.shield.risk_vnext_weight_prop_clear),
            "risk_vnext_weight_clear_gap": float(shield.cfg.shield.risk_vnext_weight_clear_gap),
            "risk_vnext_weight_support": float(shield.cfg.shield.risk_vnext_weight_support),
            "risk_vnext_weight_region": float(shield.cfg.shield.risk_vnext_weight_region),
            "risk_vnext2_weight_prop_clear": float(shield.cfg.shield.risk_vnext2_weight_prop_clear),
            "risk_vnext2_weight_fragility": float(shield.cfg.shield.risk_vnext2_weight_fragility),
            "risk_vnext2_weight_support": float(shield.cfg.shield.risk_vnext2_weight_support),
            "risk_vnext2_weight_region": float(shield.cfg.shield.risk_vnext2_weight_region),
            "risk_hist_window": int(shield.cfg.shield.risk_hist_window),
            "risk_threat_count_norm": float(shield.cfg.shield.risk_threat_count_norm),
        },
    }


def compute_variant_score(record: OfflineSafeRecord, variant: RiskVariant) -> tuple[float, float, float, float, float, float, float]:
    clear_min = float(record.risk_clear_v1)
    clear_prop = float(record.risk_clear_prop)
    clear_gap = float(record.risk_clear_gap)
    fragility = float(record.hard_set_fragility)
    region = float(record.risk_region)
    hist = float(record.risk_hist if variant.weight_hist != 0.0 else 0.0)
    support = float(record.risk_support)
    score = (
        float(variant.weight_clear_min) * clear_min
        + float(variant.weight_clear_prop) * clear_prop
        + float(variant.weight_clear_gap) * clear_gap
        + float(variant.weight_fragility) * fragility
        + float(variant.weight_region) * region
        + float(variant.weight_hist) * hist
        + float(variant.weight_support) * support
    )
    return float(score), clear_min, clear_prop, clear_gap, fragility, region, hist, support


def summarize_variant(records: Sequence[OfflineSafeRecord], variant: RiskVariant, thresholds: Sequence[float]) -> Dict[str, Any]:
    base_rows: List[Dict[str, float]] = []
    for record in records:
        score, clear_min, clear_prop, clear_gap, fragility, region, hist, support = compute_variant_score(record, variant)
        row = {
            **asdict(record),
            "variant": str(variant.name),
            "variant_score": float(score),
            "variant_clear_min": float(clear_min),
            "variant_clear_prop": float(clear_prop),
            "variant_clear_gap": float(clear_gap),
            "variant_fragility": float(fragility),
            "variant_region": float(region),
            "variant_hist": float(hist),
            "variant_support": float(support),
            "eligible": int(record.proposed_in_a_hard),
        }
        base_rows.append(row)

    total = max(len(base_rows), 1)
    eligible_rows = [row for row in base_rows if int(row["eligible"]) == 1]
    hard_empty_rows = [row for row in base_rows if int(row["a_hard_size"]) == 0]
    ineligible_nonempty_rows = [
        row for row in base_rows if int(row["eligible"]) == 0 and int(row["a_hard_size"]) > 0
    ]
    positives = [row for row in base_rows if int(row["need_rec"]) == 1]
    negatives = [row for row in base_rows if int(row["need_rec"]) == 0]
    eligible_positives = [row for row in eligible_rows if int(row["need_rec"]) == 1]
    eligible_negatives = [row for row in eligible_rows if int(row["need_rec"]) == 0]
    need_rec_count = len(positives)
    eligible_count = len(eligible_rows)

    def mean_key(key: str, rows: Sequence[Dict[str, float]]) -> float:
        if not rows:
            return 0.0
        return float(np.mean([float(row[key]) for row in rows]))

    threshold_scan: List[Dict[str, float]] = []
    best_threshold = float(thresholds[0])
    best_score = -1.0
    best_threshold_eligible = float(thresholds[0])
    best_score_eligible = -1.0
    for threshold in thresholds:
        predicted = [row for row in base_rows if float(row["variant_score"]) >= float(threshold)]
        tp = sum(1 for row in predicted if int(row["need_rec"]) == 1)
        pred_pos = len(predicted)
        predicted_eligible = [row for row in predicted if int(row["eligible"]) == 1]
        pred_eligible_count = len(predicted_eligible)
        wasted_gate_count = pred_pos - pred_eligible_count
        precision = float(tp / pred_pos) if pred_pos else 0.0
        recall = float(tp / need_rec_count) if need_rec_count else 0.0
        eligible_precision = float(tp / pred_eligible_count) if pred_eligible_count else 0.0
        scan = {
            "risk_threshold": float(threshold),
            "gate_rate": float(pred_pos / total),
            "eligible_gate_rate": float(pred_eligible_count / max(eligible_count, 1)),
            "wasted_gate_rate": float(wasted_gate_count / total),
            "precision_need_rec": precision,
            "eligible_precision_need_rec": eligible_precision,
            "recall_need_rec": recall,
        }
        threshold_scan.append(scan)
        score = float(scan["precision_need_rec"] + scan["recall_need_rec"])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
        score_eligible = float(scan["eligible_precision_need_rec"] + scan["recall_need_rec"])
        if score_eligible > best_score_eligible:
            best_score_eligible = score_eligible
            best_threshold_eligible = float(threshold)

    best_row = next(scan for scan in threshold_scan if float(scan["risk_threshold"]) == float(best_threshold))
    best_row_eligible = next(scan for scan in threshold_scan if float(scan["risk_threshold"]) == float(best_threshold_eligible))
    return {
        "primary_view": "eligible_only",
        "variant": variant.name,
        "description": variant.description,
        "weight_clear_min": float(variant.weight_clear_min),
        "weight_clear_prop": float(variant.weight_clear_prop),
        "weight_clear_gap": float(variant.weight_clear_gap),
        "weight_fragility": float(variant.weight_fragility),
        "weight_region": float(variant.weight_region),
        "weight_hist": float(variant.weight_hist),
        "weight_support": float(variant.weight_support),
        "agent_step_count": int(len(base_rows)),
        "eligible_agent_step_count": int(eligible_count),
        "eligible_agent_step_rate": float(eligible_count / total),
        "hard_empty_agent_step_count": int(len(hard_empty_rows)),
        "hard_empty_agent_step_rate": float(len(hard_empty_rows) / total),
        "dead_end_hard_agent_step_count": int(len(hard_empty_rows)),
        "dead_end_hard_agent_step_rate": float(len(hard_empty_rows) / total),
        "ineligible_nonempty_agent_step_count": int(len(ineligible_nonempty_rows)),
        "ineligible_nonempty_agent_step_rate": float(len(ineligible_nonempty_rows) / total),
        "need_rec_count": int(need_rec_count),
        "need_rec_rate": float(need_rec_count / total),
        "set_diff_rate": float(sum(int(row["set_diff"]) for row in base_rows) / total),
        "best_threshold_by_precision_plus_recall": float(best_threshold),
        "best_threshold_metrics": best_row,
        "best_threshold_by_eligible_precision_plus_recall": float(best_threshold_eligible),
        "best_threshold_eligible_metrics": best_row_eligible,
        "primary_best_threshold": float(best_threshold_eligible),
        "primary_best_threshold_metrics": best_row_eligible,
        "risk_score_need_rec_pos_mean": mean_key("variant_score", positives),
        "risk_score_need_rec_neg_mean": mean_key("variant_score", negatives),
        "eligible_risk_score_need_rec_pos_mean": mean_key("variant_score", eligible_positives),
        "eligible_risk_score_need_rec_neg_mean": mean_key("variant_score", eligible_negatives),
        "risk_clear_min_need_rec_pos_mean": mean_key("variant_clear_min", positives),
        "risk_clear_min_need_rec_neg_mean": mean_key("variant_clear_min", negatives),
        "eligible_risk_clear_min_need_rec_pos_mean": mean_key("variant_clear_min", eligible_positives),
        "eligible_risk_clear_min_need_rec_neg_mean": mean_key("variant_clear_min", eligible_negatives),
        "risk_clear_prop_need_rec_pos_mean": mean_key("variant_clear_prop", positives),
        "risk_clear_prop_need_rec_neg_mean": mean_key("variant_clear_prop", negatives),
        "eligible_risk_clear_prop_need_rec_pos_mean": mean_key("variant_clear_prop", eligible_positives),
        "eligible_risk_clear_prop_need_rec_neg_mean": mean_key("variant_clear_prop", eligible_negatives),
        "risk_clear_gap_need_rec_pos_mean": mean_key("variant_clear_gap", positives),
        "risk_clear_gap_need_rec_neg_mean": mean_key("variant_clear_gap", negatives),
        "eligible_risk_clear_gap_need_rec_pos_mean": mean_key("variant_clear_gap", eligible_positives),
        "eligible_risk_clear_gap_need_rec_neg_mean": mean_key("variant_clear_gap", eligible_negatives),
        "risk_fragility_need_rec_pos_mean": mean_key("variant_fragility", positives),
        "risk_fragility_need_rec_neg_mean": mean_key("variant_fragility", negatives),
        "eligible_risk_fragility_need_rec_pos_mean": mean_key("variant_fragility", eligible_positives),
        "eligible_risk_fragility_need_rec_neg_mean": mean_key("variant_fragility", eligible_negatives),
        "risk_region_need_rec_pos_mean": mean_key("variant_region", positives),
        "risk_region_need_rec_neg_mean": mean_key("variant_region", negatives),
        "eligible_risk_region_need_rec_pos_mean": mean_key("variant_region", eligible_positives),
        "eligible_risk_region_need_rec_neg_mean": mean_key("variant_region", eligible_negatives),
        "risk_hist_need_rec_pos_mean": mean_key("variant_hist", positives),
        "risk_hist_need_rec_neg_mean": mean_key("variant_hist", negatives),
        "eligible_risk_hist_need_rec_pos_mean": mean_key("variant_hist", eligible_positives),
        "eligible_risk_hist_need_rec_neg_mean": mean_key("variant_hist", eligible_negatives),
        "risk_support_need_rec_pos_mean": mean_key("variant_support", positives),
        "risk_support_need_rec_neg_mean": mean_key("variant_support", negatives),
        "eligible_risk_support_need_rec_pos_mean": mean_key("variant_support", eligible_positives),
        "eligible_risk_support_need_rec_neg_mean": mean_key("variant_support", eligible_negatives),
        "threshold_scan": threshold_scan,
        "rows": base_rows,
    }


def maybe_write_records_csv(path: str | None, rows: Sequence[Dict[str, Any]]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline risk validation on a fixed safe trajectory with hist ablation and proposed-action-clearance v2."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path for fixed-policy evaluation.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.35, 0.5, 0.65, 0.8],
        help="Risk thresholds to scan on the fixed safe trajectory.",
    )
    parser.add_argument("--map-size", type=int, default=None)
    parser.add_argument("--n-uavs", type=int, default=None)
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--n-threats", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--records-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_overrides = {
        "map_size": args.map_size,
        "n_uavs": args.n_uavs,
        "n_targets": args.n_targets,
        "n_threats": args.n_threats,
        "max_steps": args.max_steps,
        "terminate_on_all_targets_found": None,
        "seed": args.seed,
    }
    thresholds = [float(v) for v in args.thresholds]

    rollout = collect_safe_trajectory_records(
        args.checkpoint,
        episodes=args.episodes,
        device=args.device,
        env_overrides=env_overrides,
    )
    records: List[OfflineSafeRecord] = rollout["records"]
    weights = rollout["shield_config"]
    variants = [
        RiskVariant(
            name="baseline_v1",
            description="Current v1 clear(min_candidate) + region + hist on a fixed safe trajectory.",
            weight_clear_min=float(weights["risk_weight_clear"]),
            weight_clear_prop=0.0,
            weight_clear_gap=0.0,
            weight_fragility=0.0,
            weight_region=float(weights["risk_weight_region"]),
            weight_hist=float(weights["risk_weight_hist"]),
            weight_support=0.0,
        ),
        RiskVariant(
            name="ablation_hist0",
            description="Hist ablation with the same clear/region weights and hist term forced to zero.",
            weight_clear_min=float(weights["risk_weight_clear"]),
            weight_clear_prop=0.0,
            weight_clear_gap=0.0,
            weight_fragility=0.0,
            weight_region=float(weights["risk_weight_region"]),
            weight_hist=0.0,
            weight_support=0.0,
        ),
        RiskVariant(
            name="v2_proposed_action_clearance",
            description="Swap the clear term from min_candidate_clearance to proposed_action_clearance, keeping region + hist.",
            weight_clear_min=0.0,
            weight_clear_prop=float(weights["risk_weight_clear"]),
            weight_clear_gap=0.0,
            weight_fragility=0.0,
            weight_region=float(weights["risk_weight_region"]),
            weight_hist=float(weights["risk_weight_hist"]),
            weight_support=0.0,
        ),
        RiskVariant(
            name="v3_hybrid_clear",
            description="Hybrid v3 with clear_min + clear_prop + region, hist removed. Weights: 0.50 / 0.35 / 0.15.",
            weight_clear_min=0.50,
            weight_clear_prop=0.35,
            weight_clear_gap=0.0,
            weight_fragility=0.0,
            weight_region=0.15,
            weight_hist=0.0,
            weight_support=0.0,
        ),
        RiskVariant(
            name="v3_hybrid_clear_fragility",
            description="Hybrid v3 with clear_min + clear_prop + fragility + region, hist removed. Weights: 0.35 / 0.25 / 0.25 / 0.15.",
            weight_clear_min=0.35,
            weight_clear_prop=0.25,
            weight_clear_gap=0.0,
            weight_fragility=0.25,
            weight_region=0.15,
            weight_hist=0.0,
            weight_support=0.0,
        ),
        RiskVariant(
            name="v_next_prop_gap_support_region",
            description="v_next with proposed-action clear + proposed-vs-best clear gap + robust-support risk + region. Weights: 0.45 / 0.25 / 0.20 / 0.10.",
            weight_clear_min=0.0,
            weight_clear_prop=float(weights["risk_vnext_weight_prop_clear"]),
            weight_clear_gap=float(weights["risk_vnext_weight_clear_gap"]),
            weight_fragility=0.0,
            weight_region=float(weights["risk_vnext_weight_region"]),
            weight_hist=0.0,
            weight_support=float(weights["risk_vnext_weight_support"]),
        ),
        RiskVariant(
            name="v_next2_prop_fragility_support_region",
            description="v_next2 with proposed-action clear + hard-set fragility + robust-support risk + region. Weights: 0.45 / 0.25 / 0.20 / 0.10.",
            weight_clear_min=0.0,
            weight_clear_prop=float(weights["risk_vnext2_weight_prop_clear"]),
            weight_clear_gap=0.0,
            weight_fragility=float(weights["risk_vnext2_weight_fragility"]),
            weight_region=float(weights["risk_vnext2_weight_region"]),
            weight_hist=0.0,
            weight_support=float(weights["risk_vnext2_weight_support"]),
        ),
        RiskVariant(
            name="cand_clearmin_prop_fragility_support_region",
            description="Candidate mix with clear_min + clear_prop + fragility + support + region. Weights: 0.25 / 0.25 / 0.20 / 0.20 / 0.10.",
            weight_clear_min=0.25,
            weight_clear_prop=0.25,
            weight_clear_gap=0.0,
            weight_fragility=0.20,
            weight_region=0.10,
            weight_hist=0.0,
            weight_support=0.20,
        ),
        RiskVariant(
            name="cand_clearmin_prop_support_region",
            description="Candidate mix with clear_min + clear_prop + support + region. Weights: 0.35 / 0.35 / 0.20 / 0.10.",
            weight_clear_min=0.35,
            weight_clear_prop=0.35,
            weight_clear_gap=0.0,
            weight_fragility=0.0,
            weight_region=0.10,
            weight_hist=0.0,
            weight_support=0.20,
        ),
        RiskVariant(
            name="vnext_tune_proxy_prop_gap_region",
            description="v_next micro-tune for runtime threshold proxy: prop_clear + clear_gap + region, with support removed. Weights: 0.40 / 0.35 / 0.25.",
            weight_clear_min=0.0,
            weight_clear_prop=0.40,
            weight_clear_gap=0.35,
            weight_fragility=0.0,
            weight_region=0.25,
            weight_hist=0.0,
            weight_support=0.0,
        ),
        RiskVariant(
            name="vnext_tune_exact_prop_support",
            description="v_next micro-tune for exact top-k ranking: prop_clear + support only. Weights: 0.50 / 0.50.",
            weight_clear_min=0.0,
            weight_clear_prop=0.50,
            weight_clear_gap=0.0,
            weight_fragility=0.0,
            weight_region=0.0,
            weight_hist=0.0,
            weight_support=0.50,
        ),
        RiskVariant(
            name="vnext_tune_exact_prop_only",
            description="v_next micro-tune for exact top-k ranking: prop_clear only. Weight: 1.00.",
            weight_clear_min=0.0,
            weight_clear_prop=1.00,
            weight_clear_gap=0.0,
            weight_fragility=0.0,
            weight_region=0.0,
            weight_hist=0.0,
            weight_support=0.0,
        ),
    ]

    summaries: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for variant in variants:
        summary = summarize_variant(records, variant, thresholds)
        summaries.append({k: v for k, v in summary.items() if k != "rows"})
        csv_rows.extend(summary["rows"])
        best = summary["best_threshold_metrics"]
        print(
            f"[offline-safe] variant={variant.name} best_threshold={summary['best_threshold_by_precision_plus_recall']:.2f} "
            f"gate_rate={best['gate_rate']:.4f} precision={best['precision_need_rec']:.4f} "
            f"eligible_precision={best['eligible_precision_need_rec']:.4f} recall={best['recall_need_rec']:.4f}"
        )

    result = {
        "checkpoint": rollout["checkpoint"],
        "trajectory_mode": "safe",
        "validation_primary_view": "eligible_only",
        "episodes": int(rollout["episodes"]),
        "env_steps": int(rollout["env_steps"]),
        "wall_time_sec": float(rollout["wall_time_sec"]),
        "thresholds": thresholds,
        "shield_config": rollout["shield_config"],
        "trajectory_summary": {
            "agent_step_count": int(len(records)),
            "eligible_agent_step_count": int(sum(int(record.proposed_in_a_hard) for record in records)),
            "eligible_agent_step_rate": float(np.mean([int(record.proposed_in_a_hard) for record in records]) if records else 0.0),
            "hard_empty_agent_step_count": int(sum(int(record.a_hard_size == 0) for record in records)),
            "hard_empty_agent_step_rate": float(np.mean([int(record.a_hard_size == 0) for record in records]) if records else 0.0),
            "dead_end_hard_agent_step_count": int(sum(int(record.a_hard_size == 0) for record in records)),
            "dead_end_hard_agent_step_rate": float(np.mean([int(record.a_hard_size == 0) for record in records]) if records else 0.0),
            "ineligible_nonempty_agent_step_count": int(
                sum(int((record.proposed_in_a_hard == 0) and (record.a_hard_size > 0)) for record in records)
            ),
            "ineligible_nonempty_agent_step_rate": float(
                np.mean([int((record.proposed_in_a_hard == 0) and (record.a_hard_size > 0)) for record in records]) if records else 0.0
            ),
            "need_rec_count": int(sum(int(record.need_rec) for record in records)),
            "need_rec_rate": float(np.mean([int(record.need_rec) for record in records]) if records else 0.0),
        },
        "variant_summaries": summaries,
        "notes": [
            "The rollout trajectory is generated once in safe mode and held fixed for all threshold / variant comparisons.",
            "A_rec_oracle is still computed with full enumerate_recursive_feasible_actions under the current sequential-adjudication semantics.",
            "The proposed_action_clearance v2/v_next clear term is only activated when the proposed action survives A_hard; otherwise its clear term is set to zero because no A_hard -> A_rec upgrade is needed.",
            "Hist ablation keeps the original clear/region weights and simply forces the hist term to zero; weights are not renormalized.",
            "hard_set_fragility is defined as 1 - |A_hard| / |A_valid| and is intended to capture whether the hard-safe layer is already losing slack.",
            "v_next support risk is defined as 1 - robust_count / |A_hard|, where a hard action is treated as robust when its clearance is at least risk_support_clearance_margin.",
            "v_next2 replaces the v_next clear-gap term with hard-set fragility after the eligible-only analysis showed clear_gap was a negative contributor.",
            "The new support-mix candidates are exploratory offline variants only; they do not change the runtime shield path unless promoted later.",
            "The vnext_tune_* candidates are a focused offline micro-tuning sweep inside the v_next component family; they do not change the runtime shield path unless promoted later.",
            "Primary interpretation should now use eligible-only metrics because need_rec is only well-defined for proposed actions that already survive A_hard.",
            "A_hard-empty and ineligible-but-nonempty samples are reported as separate buckets because they are safety-relevant but semantically different from the A_hard -> A_rec upgrade problem.",
        ],
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    maybe_write_records_csv(args.records_csv, csv_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
