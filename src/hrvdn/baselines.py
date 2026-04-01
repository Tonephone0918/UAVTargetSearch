from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .env import UAV_H_DIRS, V_DIRS


def select_greedy_actions(obs: List[Dict]) -> List[int]:
    """A simple greedy baseline for visualization.

    It chooses the horizontal move whose adjacent cell has the largest local
    target probability in the TPM patch. For the vertical move, it prefers
    keeping the current altitude when valid.
    """

    actions: List[int] = []
    for o in obs:
        mask = np.asarray(o["action_mask"], dtype=bool)
        tpm_patch = np.asarray(o["map"], dtype=np.float32)[0]
        center = tpm_patch.shape[0] // 2
        n_v_actions = len(V_DIRS)

        best_h = None
        best_score = float("-inf")
        for h, (dx, dy) in enumerate(UAV_H_DIRS):
            group_mask = mask[h * n_v_actions : (h + 1) * n_v_actions]
            if not group_mask.any():
                continue

            px = int(np.clip(center + dx, 0, tpm_patch.shape[0] - 1))
            py = int(np.clip(center + dy, 0, tpm_patch.shape[1] - 1))
            score = float(tpm_patch[px, py])
            if best_h is None or score > best_score:
                best_h = h
                best_score = score

        if best_h is None:
            valid = np.flatnonzero(mask)
            actions.append(int(valid[0]) if len(valid) > 0 else 0)
            continue

        preferred_action = best_h * n_v_actions + 1
        if mask[preferred_action]:
            actions.append(preferred_action)
            continue

        group_valid = np.flatnonzero(mask[best_h * n_v_actions : (best_h + 1) * n_v_actions])
        actions.append(best_h * n_v_actions + int(group_valid[0]))

    return actions


def get_baseline_action_selector(name: str) -> Callable[[List[Dict]], List[int]]:
    if name == "greedy":
        return select_greedy_actions
    raise ValueError(f"Unsupported baseline policy: {name}")
