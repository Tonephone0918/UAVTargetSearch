from __future__ import annotations

from time import perf_counter
from typing import Dict

import numpy as np
import torch

from .shield import CentralizedSafetyShield
from .stats import EpisodeStatsAccumulator


def _finalize_metrics(metrics: Dict[str, list[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in metrics.items()}


@torch.no_grad()
def evaluate(
    env,
    policy,
    episodes: int = 5,
    device: str = "cpu",
    shield: CentralizedSafetyShield | None = None,
) -> Dict[str, float]:
    metrics: Dict[str, list[float]] = {}
    for _ in range(episodes):
        obs = env.reset()
        if shield is not None:
            shield.reset_episode()
        done = False
        hs = [torch.zeros(1, 1, policy.gru.hidden_size, device=device) for _ in range(env.cfg.n_uavs)]
        episode_stats = EpisodeStatsAccumulator()
        last_info = None
        episode_step_time = 0.0
        episode_stats_time = 0.0
        while not done:
            step_start = perf_counter()
            acts = []
            preferences = []
            for i, o in enumerate(obs):
                om = torch.tensor(o["map"], dtype=torch.float32, device=device).flatten().unsqueeze(0)
                ex = torch.tensor(o["extra"], dtype=torch.float32, device=device).unsqueeze(0)
                q, hs[i] = policy(om, ex, hs[i])
                mask = torch.tensor(o["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
                q = q.masked_fill(~mask, -1e9)
                preferences.append(q.squeeze(0))
                acts.append(int(q.argmax(dim=-1).item()))
            shield_stats = {"shield_penalty": 0.0}
            if shield is not None:
                acts, _, shield_stats = shield.apply(
                    env,
                    acts,
                    torch.stack(preferences, dim=0),
                    np.stack([o["action_mask"] for o in obs]),
                    selection_mode="argmax",
                )
            obs, r, done, info = env.step(acts)
            if shield is not None:
                r, info = shield.apply_reward_penalty(r, info, shield_stats)
            else:
                info = dict(info)
                info["original_reward"] = float(r)
                info["shield_penalized_reward"] = float(r)
            stats_start = perf_counter()
            episode_stats.update(info)
            episode_stats_time += perf_counter() - stats_start
            last_info = info
            episode_step_time += perf_counter() - step_start
        episode_metrics = episode_stats.finalize(
            last_info or {},
            found_targets=len(env.found_targets),
            shield_mode=shield.cfg.shield.mode if shield is not None else "off",
        )
        if shield is not None:
            episode_metrics.update(shield.profile_summary(episode_step_time, episode_stats_time))
        for key, value in episode_metrics.items():
            metrics.setdefault(key, []).append(float(value))

    return _finalize_metrics(metrics)


@torch.no_grad()
def evaluate_actor_policy(
    env,
    actor,
    episodes: int = 5,
    device: str = "cpu",
    shield: CentralizedSafetyShield | None = None,
) -> Dict[str, float]:
    metrics: Dict[str, list[float]] = {}
    for _ in range(episodes):
        obs = env.reset()
        if shield is not None:
            shield.reset_episode()
        done = False
        episode_stats = EpisodeStatsAccumulator()
        last_info = None
        episode_step_time = 0.0
        episode_stats_time = 0.0
        while not done:
            step_start = perf_counter()
            maps = torch.tensor(np.stack([o["map"] for o in obs]), dtype=torch.float32, device=device)
            extras = torch.tensor(np.stack([o["extra"] for o in obs]), dtype=torch.float32, device=device)
            masks = torch.tensor(np.stack([o["action_mask"] for o in obs]), dtype=torch.bool, device=device)
            logits = actor(maps, extras)
            logits = logits.masked_fill(~masks, -1e9)
            acts = logits.argmax(dim=-1).tolist()
            shield_stats = {"shield_penalty": 0.0}
            if shield is not None:
                acts, _, shield_stats = shield.apply(
                    env,
                    acts,
                    logits,
                    masks.detach().cpu().numpy(),
                    selection_mode="argmax",
                )
            obs, r, done, info = env.step(acts)
            if shield is not None:
                r, info = shield.apply_reward_penalty(r, info, shield_stats)
            else:
                info = dict(info)
                info["original_reward"] = float(r)
                info["shield_penalized_reward"] = float(r)
            stats_start = perf_counter()
            episode_stats.update(info)
            episode_stats_time += perf_counter() - stats_start
            last_info = info
            episode_step_time += perf_counter() - step_start
        episode_metrics = episode_stats.finalize(
            last_info or {},
            found_targets=len(env.found_targets),
            shield_mode=shield.cfg.shield.mode if shield is not None else "off",
        )
        if shield is not None:
            episode_metrics.update(shield.profile_summary(episode_step_time, episode_stats_time))
        for key, value in episode_metrics.items():
            metrics.setdefault(key, []).append(float(value))
    return _finalize_metrics(metrics)
