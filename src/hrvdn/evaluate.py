from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.distributions import Categorical


def _finalize_metrics(metrics: Dict[str, list[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in metrics.items()}


@torch.no_grad()
def evaluate(env, policy, episodes: int = 5, device: str = "cpu") -> Dict[str, float]:
    metrics = {"search_rate": [], "coverage_rate": [], "collisions": [], "avg_reward": [], "error_rate": []}
    for _ in range(episodes):
        obs = env.reset()
        done = False
        hs = [torch.zeros(1, 1, policy.gru.hidden_size, device=device) for _ in range(env.cfg.n_uavs)]
        ep_reward = 0.0
        last_info = None
        while not done:
            acts = []
            for i, o in enumerate(obs):
                om = torch.tensor(o["map"], dtype=torch.float32, device=device).flatten().unsqueeze(0)
                ex = torch.tensor(o["extra"], dtype=torch.float32, device=device).unsqueeze(0)
                q, hs[i] = policy(om, ex, hs[i])
                mask = torch.tensor(o["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
                q = q.masked_fill(~mask, -1e9)
                acts.append(int(q.argmax(dim=-1).item()))
            obs, r, done, info = env.step(acts)
            ep_reward += r
            last_info = info
        metrics["search_rate"].append(last_info["search_rate"])
        metrics["coverage_rate"].append(last_info["coverage_rate"])
        metrics["collisions"].append(env.collisions)
        metrics["avg_reward"].append(ep_reward)
        metrics["error_rate"].append(last_info["error_rate"])

    return _finalize_metrics(metrics)


@torch.no_grad()
def evaluate_actor_policy(env, actor, episodes: int = 5, device: str = "cpu") -> Dict[str, float]:
    metrics = {"search_rate": [], "coverage_rate": [], "collisions": [], "avg_reward": [], "error_rate": []}
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        last_info = None
        while not done:
            maps = torch.tensor(np.stack([o["map"] for o in obs]), dtype=torch.float32, device=device)
            extras = torch.tensor(np.stack([o["extra"] for o in obs]), dtype=torch.float32, device=device)
            masks = torch.tensor(np.stack([o["action_mask"] for o in obs]), dtype=torch.bool, device=device)
            logits = actor(maps, extras)
            logits = logits.masked_fill(~masks, -1e9)
            dist = Categorical(logits=logits)
            acts = dist.probs.argmax(dim=-1).tolist()
            obs, r, done, info = env.step(acts)
            ep_reward += r
            last_info = info
        metrics["search_rate"].append(last_info["search_rate"])
        metrics["coverage_rate"].append(last_info["coverage_rate"])
        metrics["collisions"].append(env.collisions)
        metrics["avg_reward"].append(ep_reward)
        metrics["error_rate"].append(last_info["error_rate"])
    return _finalize_metrics(metrics)
