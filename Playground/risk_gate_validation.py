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
from src.hrvdn.validate import evaluate_checkpoint


@dataclass
class StepAgentRecord:
    episode: int
    step: int
    agent_idx: int
    risk_score: float
    risk_clear: float
    risk_clear_gap: float
    risk_fragility: float
    risk_region: float
    risk_hist: float
    risk_support: float
    a_hard_size: int
    a_rec_runtime_size: int
    a_rec_oracle_size: int
    proposed_action: int
    final_action: int
    proposed_in_a_hard: int
    proposed_in_a_rec_runtime: int
    proposed_in_a_rec_oracle: int
    runtime_recursive_gate: int
    high_risk: int
    need_rec: int
    set_diff: int
    action_replaced: int
    fallback_triggered: int
    used_legacy_gate: int


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
    shield_overrides: Dict[str, Any] | None = None,
) -> tuple[str, UAVSearchEnv, PolicyRunner, CentralizedSafetyShield]:
    run_device = resolve_device(device)
    ckpt = torch.load(checkpoint_path, map_location=run_device)
    algo = ckpt.get("algo", "hrvdn")
    cfg = config_from_dict(ckpt.get("config", {}))
    cfg.reward.mode = ckpt.get("reward_mode", cfg.reward.mode)
    apply_env_overrides(cfg, **(env_overrides or {}))
    for key, value in (shield_overrides or {}).items():
        if hasattr(cfg.shield, key):
            setattr(cfg.shield, key, value)
    if cfg.shield.mode == "off":
        cfg.shield.enabled = False
    elif cfg.shield.enabled:
        cfg.shield.mode = cfg.shield.mode or "safe"
    else:
        cfg.shield.enabled = True
        cfg.shield.mode = cfg.shield.mode or "safe"

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
    return algo, env, runner, shield


def _sorted_unique_ints(values: Iterable[int]) -> List[int]:
    return sorted({int(v) for v in values})


def apply_with_oracle_diagnostics(
    env: UAVSearchEnv,
    shield: CentralizedSafetyShield,
    proposed_actions: Sequence[int],
    actor_preferences: torch.Tensor | np.ndarray,
    action_masks: Sequence[Sequence[bool]] | np.ndarray,
    *,
    selection_mode: str,
    episode_idx: int,
    step_idx: int,
) -> tuple[List[int], List[StepAgentRecord]]:
    proposed = [int(a) for a in proposed_actions]
    valid_masks = np.asarray(action_masks, dtype=bool)
    state = shield._capture_state(env)
    final_actions = list(proposed)
    effective_masks = valid_masks.copy()
    planned_next_positions = shield._planned_next_positions(state, final_actions)
    records: List[StepAgentRecord] = []

    shield_triggered = False
    triggered_agents = 0
    fallback_triggered_step = False

    for agent_idx in range(shield.cfg.env.n_uavs):
        base_actions = list(final_actions)
        hard_actions, hard_meta = shield._enumerate_hard_actions_with_meta(
            env,
            state,
            agent_idx,
            base_actions,
            planned_next_positions=planned_next_positions,
        )
        proposed_action = int(final_actions[agent_idx])
        risk_info = shield.compute_agent_risk(
            agent_idx,
            hard_actions,
            hard_meta,
            proposed_action=proposed_action,
        )

        if hard_actions:
            oracle_rec_actions = shield.enumerate_recursive_feasible_actions(
                env,
                state,
                agent_idx,
                base_actions,
                hard_actions,
            )
        else:
            oracle_rec_actions = []
        oracle_rec_actions = _sorted_unique_ints(oracle_rec_actions)

        runtime_rec_actions: List[int]
        used_legacy_gate = int(shield._uses_legacy_recursive_gate())
        runtime_gate = 0
        if shield.cfg.shield.mode == "recursive":
            runtime_rec_actions, decision_meta = shield._decision_recursive_actions(
                env,
                state,
                agent_idx,
                base_actions,
                hard_actions,
                proposed_action,
                actor_preferences[agent_idx],
                hard_meta,
                risk_info,
            )
            runtime_rec_actions = _sorted_unique_ints(runtime_rec_actions)
            runtime_gate = int(bool(decision_meta.get("recursive_gate_run", False)))
        else:
            runtime_rec_actions = _sorted_unique_ints(hard_actions)

        candidate_actions = hard_actions
        if shield.cfg.shield.mode == "recursive":
            candidate_actions = runtime_rec_actions if runtime_rec_actions else hard_actions

        proposed_in_a_hard = int(proposed_action in hard_actions)
        proposed_in_a_rec_runtime = int(proposed_action in runtime_rec_actions)
        proposed_in_a_rec_oracle = int(proposed_action in oracle_rec_actions)
        need_rec = int(bool(proposed_in_a_hard and not proposed_in_a_rec_oracle))
        set_diff = int(_sorted_unique_ints(hard_actions) != oracle_rec_actions)

        proposed_is_admissible = proposed_action in candidate_actions if candidate_actions else False
        if shield.cfg.shield.mode == "safe":
            proposed_is_admissible = proposed_action in hard_actions

        selected_action = proposed_action
        agent_intervened = False
        fallback_triggered_agent = 0
        if proposed_is_admissible:
            shield.agent_intervention_history[agent_idx].append(0)
        else:
            shield_triggered = True
            triggered_agents += 1
            if candidate_actions:
                selected_action, selected_mask = shield.resample_action_from_allowed_set(
                    actor_preferences[agent_idx],
                    candidate_actions,
                    valid_masks[agent_idx],
                    selection_mode=selection_mode,
                )
                effective_masks[agent_idx] = selected_mask
            else:
                fallback_triggered_step = True
                fallback_triggered_agent = 1
                shield.fallback_count += 1
                fallback_candidates = np.flatnonzero(valid_masks[agent_idx]).tolist()
                if fallback_candidates:
                    selected_action, selected_mask = shield.resample_action_from_allowed_set(
                        actor_preferences[agent_idx],
                        fallback_candidates,
                        valid_masks[agent_idx],
                        selection_mode=selection_mode,
                    )
                    effective_masks[agent_idx] = selected_mask
                else:
                    selected_action = proposed_action
            if int(selected_action) != proposed_action:
                agent_intervened = True
                shield.action_replaced_count += 1
            final_actions[agent_idx] = int(selected_action)
            planned_next_positions[agent_idx] = shield._single_next_position(state, agent_idx, int(selected_action))
            shield.agent_intervention_history[agent_idx].append(1 if agent_intervened else 0)

        records.append(
            StepAgentRecord(
                episode=int(episode_idx),
                step=int(step_idx),
                agent_idx=int(agent_idx),
                risk_score=float(risk_info["score"]),
                risk_clear=float(risk_info["clear"]),
                risk_clear_gap=float(risk_info.get("clear_gap", 0.0)),
                risk_fragility=float(risk_info.get("fragility", 0.0)),
                risk_region=float(risk_info["region"]),
                risk_hist=float(risk_info["hist"]),
                risk_support=float(risk_info.get("support", 0.0)),
                a_hard_size=int(len(hard_actions)),
                a_rec_runtime_size=int(len(runtime_rec_actions)),
                a_rec_oracle_size=int(len(oracle_rec_actions)),
                proposed_action=int(proposed_action),
                final_action=int(selected_action),
                proposed_in_a_hard=int(proposed_in_a_hard),
                proposed_in_a_rec_runtime=int(proposed_in_a_rec_runtime),
                proposed_in_a_rec_oracle=int(proposed_in_a_rec_oracle),
                runtime_recursive_gate=int(runtime_gate),
                high_risk=int(bool(risk_info.get("high_risk", False))),
                need_rec=int(need_rec),
                set_diff=int(set_diff),
                action_replaced=int(agent_intervened),
                fallback_triggered=int(fallback_triggered_agent),
                used_legacy_gate=int(used_legacy_gate),
            )
        )

    if shield_triggered:
        shield.shield_trigger_count += 1
        shield.shield_agent_trigger_count += triggered_agents
    shield.recent_trigger_history.append(1 if shield_triggered else 0)
    _ = effective_masks
    return final_actions, records


def summarize_records(records: Sequence[StepAgentRecord], thresholds: Sequence[float]) -> Dict[str, Any]:
    rows = [asdict(r) for r in records]
    total = max(len(rows), 1)
    need_rec_count = int(sum(int(row["need_rec"]) for row in rows))
    set_diff_count = int(sum(int(row["set_diff"]) for row in rows))

    def mean_key(key: str, *, subset: Sequence[Dict[str, Any]] | None = None) -> float:
        source = rows if subset is None else subset
        if not source:
            return 0.0
        return float(np.mean([float(row[key]) for row in source]))

    positives = [row for row in rows if int(row["need_rec"]) == 1]
    negatives = [row for row in rows if int(row["need_rec"]) == 0]

    threshold_scan: List[Dict[str, float]] = []
    for threshold in thresholds:
        predicted = [row for row in rows if float(row["risk_score"]) >= float(threshold)]
        tp = sum(1 for row in predicted if int(row["need_rec"]) == 1)
        pred_pos = len(predicted)
        precision = float(tp / pred_pos) if pred_pos else 0.0
        recall = float(tp / need_rec_count) if need_rec_count else 0.0
        threshold_scan.append(
            {
                "risk_threshold": float(threshold),
                "gate_rate": float(pred_pos / total),
                "precision_need_rec": precision,
                "recall_need_rec": recall,
            }
        )

    return {
        "agent_step_count": int(len(rows)),
        "need_rec_count": int(need_rec_count),
        "need_rec_rate": float(need_rec_count / total),
        "set_diff_count": int(set_diff_count),
        "set_diff_rate": float(set_diff_count / total),
        "runtime_recursive_gate_rate": mean_key("runtime_recursive_gate"),
        "runtime_high_risk_rate": mean_key("high_risk"),
        "risk_score_mean": mean_key("risk_score"),
        "risk_score_need_rec_pos_mean": mean_key("risk_score", subset=positives),
        "risk_score_need_rec_neg_mean": mean_key("risk_score", subset=negatives),
        "risk_clear_need_rec_pos_mean": mean_key("risk_clear", subset=positives),
        "risk_clear_need_rec_neg_mean": mean_key("risk_clear", subset=negatives),
        "risk_clear_gap_need_rec_pos_mean": mean_key("risk_clear_gap", subset=positives),
        "risk_clear_gap_need_rec_neg_mean": mean_key("risk_clear_gap", subset=negatives),
        "risk_fragility_need_rec_pos_mean": mean_key("risk_fragility", subset=positives),
        "risk_fragility_need_rec_neg_mean": mean_key("risk_fragility", subset=negatives),
        "risk_region_need_rec_pos_mean": mean_key("risk_region", subset=positives),
        "risk_region_need_rec_neg_mean": mean_key("risk_region", subset=negatives),
        "risk_hist_need_rec_pos_mean": mean_key("risk_hist", subset=positives),
        "risk_hist_need_rec_neg_mean": mean_key("risk_hist", subset=negatives),
        "risk_support_need_rec_pos_mean": mean_key("risk_support", subset=positives),
        "risk_support_need_rec_neg_mean": mean_key("risk_support", subset=negatives),
        "a_hard_size_mean": mean_key("a_hard_size"),
        "a_rec_oracle_size_mean": mean_key("a_rec_oracle_size"),
        "proposed_in_a_hard_rate": mean_key("proposed_in_a_hard"),
        "proposed_in_a_rec_oracle_rate": mean_key("proposed_in_a_rec_oracle"),
        "threshold_scan": threshold_scan,
    }


def run_diagnostic_rollout(
    checkpoint_path: str,
    *,
    episodes: int,
    device: str,
    env_overrides: Dict[str, Any] | None,
    shield_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    _, env, runner, shield = load_eval_stack(
        checkpoint_path,
        device=device,
        env_overrides=env_overrides,
        shield_overrides=shield_overrides,
    )
    records: List[StepAgentRecord] = []
    total_steps = 0
    wall_start = perf_counter()
    for episode_idx in range(episodes):
        obs = env.reset()
        runner.reset_episode()
        shield.reset_episode()
        done = False
        step_idx = 0
        while not done:
            proposed, preferences, valid_masks = runner.propose(obs)
            final_actions, step_records = apply_with_oracle_diagnostics(
                env,
                shield,
                proposed,
                preferences,
                valid_masks,
                selection_mode="argmax",
                episode_idx=episode_idx,
                step_idx=step_idx,
            )
            obs, _, done, _ = env.step(final_actions)
            records.extend(step_records)
            step_idx += 1
            total_steps += 1
    return {
        "episodes": int(episodes),
        "env_steps": int(total_steps),
        "wall_time_sec": float(perf_counter() - wall_start),
        "records": records,
    }


def run_threshold_performance_scan(
    checkpoint_path: str,
    *,
    episodes: int,
    device: str,
    env_overrides: Dict[str, Any] | None,
    thresholds: Sequence[float],
    risk_variant: str,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for threshold in thresholds:
        metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            episodes=episodes,
            device=device,
            env_overrides=env_overrides,
            shield_overrides={
                "enabled": True,
                "mode": "recursive",
                "profile_enabled": True,
                "risk_score_enabled": True,
                "risk_variant": str(risk_variant),
                "legacy_recursive_gate": False,
                "risk_threshold": float(threshold),
            },
        )
        results.append(
            {
                "risk_threshold": float(threshold),
                "recursive_gate_rate": float(metrics.get("recursive_gate_rate", 0.0)),
                "perf_recursive_time_ms": float(metrics.get("perf_recursive_time_ms", 0.0)),
                "perf_shield_time_ms": float(metrics.get("perf_shield_time_ms", 0.0)),
                "perf_steps_per_sec": float(metrics.get("perf_steps_per_sec", 0.0)),
            }
        )
    return results


def run_mode_comparison(
    checkpoint_path: str,
    *,
    episodes: int,
    device: str,
    env_overrides: Dict[str, Any] | None,
    risk_threshold: float,
    risk_variant: str,
) -> Dict[str, Dict[str, float]]:
    mode_settings = {
        "safe": {
            "enabled": True,
            "mode": "safe",
            "profile_enabled": True,
        },
        "recursive_legacy_gate": {
            "enabled": True,
            "mode": "recursive",
            "profile_enabled": True,
            "legacy_recursive_gate": True,
        },
        "recursive_risk_gate": {
            "enabled": True,
            "mode": "recursive",
            "profile_enabled": True,
            "legacy_recursive_gate": False,
            "risk_score_enabled": True,
            "risk_variant": str(risk_variant),
            "risk_threshold": float(risk_threshold),
        },
        "recursive_always": {
            "enabled": True,
            "mode": "recursive",
            "profile_enabled": True,
            "legacy_recursive_gate": False,
            "risk_score_enabled": True,
            "risk_variant": str(risk_variant),
            "risk_threshold": 0.0,
        },
    }
    results: Dict[str, Dict[str, float]] = {}
    for name, overrides in mode_settings.items():
        metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            episodes=episodes,
            device=device,
            env_overrides=env_overrides,
            shield_overrides=overrides,
        )
        results[name] = {
            "recursive_gate_rate": float(metrics.get("recursive_gate_rate", 0.0)),
            "dead_end_rec_rate": float(metrics.get("dead_end_rec_rate", 0.0)),
            "action_replacement_rate": float(metrics.get("action_replacement_rate", 0.0)),
            "avg_rec_action_count": float(metrics.get("avg_rec_action_count", 0.0)),
            "perf_recursive_time_ms": float(metrics.get("perf_recursive_time_ms", 0.0)),
            "perf_steps_per_sec": float(metrics.get("perf_steps_per_sec", 0.0)),
        }
    return results


def maybe_write_records_csv(path: str | None, records: Sequence[StepAgentRecord]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else list(StepAgentRecord.__annotations__.keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate whether the configured risk score is useful for gating A_hard -> A_rec."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path for fixed-policy evaluation.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--diagnostic-episodes", type=int, default=2)
    parser.add_argument("--perf-episodes", type=int, default=3)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.35, 0.5, 0.65, 0.8],
        help="Risk thresholds to scan.",
    )
    parser.add_argument("--map-size", type=int, default=None)
    parser.add_argument("--n-uavs", type=int, default=None)
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--n-threats", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--risk-variant",
        choices=["v1", "v_next", "v_next2"],
        default="v1",
        help="Risk variant used by the runtime recursive gate.",
    )
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

    diagnostic_by_threshold: List[Dict[str, Any]] = []
    all_records: List[StepAgentRecord] = []
    for threshold in thresholds:
        rollout = run_diagnostic_rollout(
            args.checkpoint,
            episodes=args.diagnostic_episodes,
            device=args.device,
            env_overrides=env_overrides,
            shield_overrides={
                "enabled": True,
                "mode": "recursive",
                "risk_score_enabled": True,
                "risk_variant": args.risk_variant,
                "legacy_recursive_gate": False,
                "risk_threshold": float(threshold),
            },
        )
        records = rollout["records"]
        summary = summarize_records(records, [threshold])
        diagnostic_by_threshold.append(
            {
                "risk_threshold": float(threshold),
                "episodes": int(rollout["episodes"]),
                "env_steps": int(rollout["env_steps"]),
                "wall_time_sec": float(rollout["wall_time_sec"]),
                **summary,
            }
        )
        all_records.extend(records)
        print(
            f"[diag] threshold={threshold:.2f} agent_steps={summary['agent_step_count']} "
            f"need_rec_rate={summary['need_rec_rate']:.4f} gate_rate={summary['threshold_scan'][0]['gate_rate']:.4f} "
            f"precision={summary['threshold_scan'][0]['precision_need_rec']:.4f} "
            f"recall={summary['threshold_scan'][0]['recall_need_rec']:.4f}"
        )

    perf_scan = run_threshold_performance_scan(
        args.checkpoint,
        episodes=args.perf_episodes,
        device=args.device,
        env_overrides=env_overrides,
        thresholds=thresholds,
        risk_variant=args.risk_variant,
    )
    best_threshold = thresholds[0]
    best_score = -1.0
    for row in diagnostic_by_threshold:
        scan = row["threshold_scan"][0]
        score = float(scan["precision_need_rec"] + scan["recall_need_rec"])
        if score > best_score:
            best_score = score
            best_threshold = float(row["risk_threshold"])

    mode_comparison = run_mode_comparison(
        args.checkpoint,
        episodes=args.perf_episodes,
        device=args.device,
        env_overrides=env_overrides,
        risk_threshold=best_threshold,
        risk_variant=args.risk_variant,
    )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": args.device,
        "diagnostic_episodes": int(args.diagnostic_episodes),
        "perf_episodes": int(args.perf_episodes),
        "risk_variant": args.risk_variant,
        "thresholds": thresholds,
        "best_threshold_by_precision_plus_recall": float(best_threshold),
        "diagnostic_by_threshold": diagnostic_by_threshold,
        "performance_by_threshold": perf_scan,
        "mode_comparison": mode_comparison,
        "notes": [
            "A_hard is always-on; risk is evaluated post-A_hard / pre-A_rec.",
            "Oracle need_rec uses full enumerate_recursive_feasible_actions on the current sequential-adjudication base actions.",
            "Runtime recursive actions may still be candidate-pruned, so oracle A_rec is stronger than the runtime top-k checked subset when candidate_full_fallback is disabled.",
            "Fallback-to-valid-mask remains in the implementation and can contaminate strong safety claims.",
        ],
    }

    maybe_write_records_csv(args.records_csv, all_records)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
