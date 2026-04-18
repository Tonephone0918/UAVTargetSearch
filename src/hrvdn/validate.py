from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from .evaluate import evaluate, evaluate_actor_policy
from .env import UAVSearchEnv
from .runtime import (
    apply_env_overrides,
    build_mappo_from_env,
    build_policy_from_env,
    config_from_dict,
    load_checkpoint_module,
    load_checkpoint_policy,
    resolve_device,
)
from .shield import CentralizedSafetyShield


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    episodes: int = 10,
    device: str = "auto",
    env_overrides: Dict | None = None,
    shield_overrides: Dict | None = None,
) -> Dict[str, float]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_device = resolve_device(device)
    ckpt = torch.load(ckpt_path, map_location=run_device)
    algo = ckpt.get("algo", "hrvdn")
    cfg_raw = ckpt.get("config", {})
    cfg = config_from_dict(cfg_raw)
    cfg.reward.mode = ckpt.get("reward_mode", cfg.reward.mode)
    apply_env_overrides(cfg, **(env_overrides or {}))
    for key, value in (shield_overrides or {}).items():
        if hasattr(cfg.shield, key):
            setattr(cfg.shield, key, value)

    env = UAVSearchEnv(cfg.env, cfg.reward, seed=cfg.train.seed)
    shield = CentralizedSafetyShield(cfg)
    if algo == "mappo":
        actor, critic = build_mappo_from_env(cfg, env, run_device)
        load_checkpoint_module(actor, ckpt["actor_state_dict"], checkpoint_path, "MAPPO actor")
        actor.eval()
        critic.eval()
        return evaluate_actor_policy(env, actor, episodes=episodes, device=run_device, shield=shield)

    policy = build_policy_from_env(cfg, env, run_device)
    load_checkpoint_policy(policy, ckpt["policy_state_dict"], checkpoint_path)
    policy.eval()
    return evaluate(env, policy, episodes=episodes, device=run_device, shield=shield)


def format_metrics(metrics: Dict[str, float]) -> str:
    order = [
        "search_rate",
        "found_targets",
        "coverage_ratio",
        "collision_count",
        "episode_return",
        "original_episode_return",
        "shield_trigger_rate",
        "action_replacement_rate",
        "avg_risk_score",
        "avg_risk_clear",
        "avg_risk_clear_gap",
        "avg_risk_fragility",
        "avg_risk_region",
        "avg_risk_hist",
        "avg_risk_support",
        "high_risk_rate",
        "recursive_gate_rate",
        "near_miss_rate",
        "avg_hard_action_count",
        "avg_safe_action_count",
        "avg_rec_action_count",
        "dead_end_hard_rate",
        "dead_end_safe_rate",
        "dead_end_rec_rate",
        "shield_penalty_rate",
        "error_rate",
        "coverage_rate",
        "collisions",
        "avg_reward",
    ]
    parts = []
    for k in order:
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.6f}")
    for k, v in metrics.items():
        if k not in order:
            parts.append(f"{k}={v:.6f}")
    return ", ".join(parts)
