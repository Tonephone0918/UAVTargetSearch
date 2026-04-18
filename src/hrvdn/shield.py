from __future__ import annotations

import math
from collections import deque
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from .config import ExperimentConfig, canonicalize_risk_variant
from .env import UAV_H_DIRS, UAVSearchEnv, V_DIRS


UAV_H_DIRS_ARRAY = np.asarray(UAV_H_DIRS, dtype=np.int32)
V_DIRS_ARRAY = np.asarray(V_DIRS, dtype=np.int32)
MAX_HORIZ_STEP = float(math.sqrt(2.0))


class CentralizedSafetyShield:
    """Centralized pre-execution shield with sequential agent adjudication.

    The runtime path is optimized for training:
    - the always-on A_hard layer uses a cheap rule-based hard-safe mask generator
    - recursive-feasible checks are conditionally triggered
    - candidate actions are pruned before expensive future-safe checks
    - lightweight caches are used for repeated local computations
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.use_legacy_path_for_profile = False
        self.reset_episode()

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.shield.enabled and self.cfg.shield.mode != "off")

    @property
    def profile_enabled(self) -> bool:
        return bool(self.cfg.shield.profile_enabled)

    @property
    def cache_enabled(self) -> bool:
        return bool(self.cfg.shield.cache_enabled)

    @property
    def risk_score_enabled(self) -> bool:
        return bool(self.cfg.shield.risk_score_enabled)

    def reset_episode(self) -> None:
        self.shield_trigger_count = 0
        self.shield_agent_trigger_count = 0
        self.action_replaced_count = 0
        self.fallback_count = 0
        self.recent_trigger_history = deque(maxlen=max(1, int(self.cfg.shield.recursive_recent_window)))
        self.agent_intervention_history = [
            deque(maxlen=max(1, int(self.cfg.shield.risk_hist_window))) for _ in range(self.cfg.env.n_uavs)
        ]
        self.predict_cache: Dict[tuple, Dict[str, Any]] = {}
        self.hard_cache: Dict[tuple, tuple[List[int], Dict[str, Any]]] = {}
        self.future_safe_cache: Dict[tuple, bool] = {}
        self.profile: Dict[str, float] = {
            "steps": 0.0,
            "shield_time": 0.0,
            "hard_time": 0.0,
            "rule_mask_time": 0.0,
            "predict_time": 0.0,
            "recursive_time": 0.0,
            "refine_time": 0.0,
            "predict_cache_queries": 0.0,
            "predict_cache_hits": 0.0,
            "hard_cache_queries": 0.0,
            "hard_cache_hits": 0.0,
            "future_cache_queries": 0.0,
            "future_cache_hits": 0.0,
            "recursive_gate_runs": 0.0,
            "recursive_gate_skips": 0.0,
            "recursive_gate_step_runs": 0.0,
            "recursive_gate_step_skips": 0.0,
            "recursive_candidate_checks": 0.0,
        }

    def profile_summary(self, total_step_time: float = 0.0, stats_time: float = 0.0) -> Dict[str, float]:
        steps = max(1.0, self.profile["steps"])
        shield_time = float(self.profile["shield_time"])
        hard_time = float(self.profile["hard_time"])
        rule_mask_time = float(self.profile["rule_mask_time"])
        predict_time = float(self.profile["predict_time"])
        recursive_time = float(self.profile["recursive_time"])
        refine_time = float(self.profile["refine_time"])
        stats_time = float(stats_time)
        total_step_time = float(total_step_time)
        return {
            "perf_step_time_ms": 1000.0 * total_step_time / steps,
            "perf_shield_time_ms": 1000.0 * shield_time / steps,
            "perf_hard_time_ms": 1000.0 * hard_time / steps,
            "perf_safe_time_ms": 1000.0 * hard_time / steps,
            "perf_rule_mask_time_ms": 1000.0 * rule_mask_time / steps,
            "perf_refine_time_ms": 1000.0 * refine_time / steps,
            "perf_predict_time_ms": 1000.0 * predict_time / steps,
            "perf_recursive_time_ms": 1000.0 * recursive_time / steps,
            "perf_stats_time_ms": 1000.0 * stats_time / steps,
            "perf_steps_per_sec": float(steps / max(total_step_time, 1e-9)),
            "perf_shield_time_ratio": float(shield_time / max(total_step_time, 1e-9)),
            "perf_predict_cache_hit_rate": float(self.profile["predict_cache_hits"] / max(self.profile["predict_cache_queries"], 1.0)),
            "perf_hard_cache_hit_rate": float(self.profile["hard_cache_hits"] / max(self.profile["hard_cache_queries"], 1.0)),
            "perf_safe_cache_hit_rate": float(self.profile["hard_cache_hits"] / max(self.profile["hard_cache_queries"], 1.0)),
            "perf_future_cache_hit_rate": float(self.profile["future_cache_hits"] / max(self.profile["future_cache_queries"], 1.0)),
            "perf_recursive_gate_run_rate": float(self.profile["recursive_gate_step_runs"] / steps),
            "perf_recursive_gate_skip_rate": float(self.profile["recursive_gate_step_skips"] / steps),
            "perf_recursive_candidate_checks": float(self.profile["recursive_candidate_checks"] / steps),
        }

    def _maybe_start(self) -> float | None:
        if not self.profile_enabled:
            return None
        return perf_counter()

    def _maybe_record(self, key: str, start: float | None) -> None:
        if start is not None:
            self.profile[key] += perf_counter() - start

    def _capture_state(self, env: UAVSearchEnv) -> Dict[str, Any]:
        return {
            "uavs": [list(u) for u in env.uavs],
            "prev_dirs": list(env.prev_dirs),
            "threats": [list(t) for t in env.threats],
        }

    def _state_key(self, state: Dict[str, Any]) -> tuple:
        cached_key = state.get("_cache_key")
        if cached_key is not None:
            return cached_key
        key = (
            tuple(tuple(int(v) for v in u) for u in state["uavs"]),
            tuple(None if d is None else (int(d[0]), int(d[1])) for d in state["prev_dirs"]),
            tuple(tuple(int(v) for v in t) for t in state["threats"]),
        )
        state["_cache_key"] = key
        return key

    def _valid_actions_for_state(self, state: Dict[str, Any], agent_idx: int) -> np.ndarray:
        s = self.cfg.env.map_size
        x, y, z = state["uavs"][agent_idx]
        prev = state["prev_dirs"][agent_idx]
        mask = np.zeros(len(UAV_H_DIRS) * len(V_DIRS), dtype=np.int8)
        fallback_angles = np.full(mask.shape[0], np.inf, dtype=np.float32)
        for a in range(mask.shape[0]):
            h, v = divmod(a, len(V_DIRS))
            dx, dy = UAV_H_DIRS[h]
            nz = z + V_DIRS[v]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < s and 0 <= ny < s):
                continue
            if not (0 <= nz < self.cfg.env.n_altitudes):
                continue
            if prev is None:
                mask[a] = 1
                fallback_angles[a] = 0.0
                continue
            dot = prev[0] * dx + prev[1] * dy
            prev_norm = math.sqrt(prev[0] ** 2 + prev[1] ** 2)
            curr_norm = math.sqrt(dx ** 2 + dy ** 2)
            ang = math.acos(np.clip(dot / (prev_norm * curr_norm + 1e-6), -1.0, 1.0))
            fallback_angles[a] = ang
            if ang <= self.cfg.env.max_turn_rad + 1e-6:
                mask[a] = 1
        if mask.sum() == 0:
            best = float(np.min(fallback_angles))
            if np.isfinite(best):
                mask[fallback_angles <= best + 1e-6] = 1
        return mask

    def predict_next_state(self, env: UAVSearchEnv, actions: Sequence[int], state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start = self._maybe_start()
        base_state = self._capture_state(env) if state is None else state
        actions_tuple = tuple(int(a) for a in actions)
        cache_key = (self._state_key(base_state), actions_tuple)
        if self.cache_enabled:
            self.profile["predict_cache_queries"] += 1.0
            cached = self.predict_cache.get(cache_key)
            if cached is not None:
                self.profile["predict_cache_hits"] += 1.0
                self._maybe_record("predict_time", start)
                return cached

        next_uavs = [list(u) for u in base_state["uavs"]]
        next_prev_dirs = list(base_state["prev_dirs"])
        for i, action in enumerate(actions_tuple):
            valid_mask = self._valid_actions_for_state(base_state, i)
            chosen = int(action)
            if valid_mask[chosen] == 0:
                chosen = int(np.flatnonzero(valid_mask)[0])
            h, v = divmod(chosen, len(V_DIRS))
            dx, dy = UAV_H_DIRS[h]
            x, y, z = next_uavs[i]
            x = int(np.clip(x + dx, 0, self.cfg.env.map_size - 1))
            y = int(np.clip(y + dy, 0, self.cfg.env.map_size - 1))
            z = int(np.clip(z + V_DIRS[v], 0, self.cfg.env.n_altitudes - 1))
            next_uavs[i] = [x, y, z]
            next_prev_dirs[i] = (dx, dy)
        out = {
            "uavs": next_uavs,
            "prev_dirs": next_prev_dirs,
            "threats": [list(t) for t in base_state["threats"]],
        }
        if self.cache_enabled:
            self.predict_cache[cache_key] = out
        self._maybe_record("predict_time", start)
        return out

    def check_hard_constraints(self, state: Dict[str, Any]) -> Dict[str, Any]:
        s = self.cfg.env.map_size
        uavs = state["uavs"]
        threats = state["threats"]
        boundary_ok = True
        uav_margins: List[float] = []
        threat_margins: List[float] = []
        for x, y, z in uavs:
            if not (0 <= x < s and 0 <= y < s and 0 <= z < self.cfg.env.n_altitudes):
                boundary_ok = False
        for i, (xi, yi, _) in enumerate(uavs):
            for j in range(i + 1, len(uavs)):
                xj, yj, _ = uavs[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                uav_margins.append(dist - self.cfg.env.uav_safe_dist)
            for tx, ty in threats:
                dist = math.sqrt((xi - tx) ** 2 + (yi - ty) ** 2)
                threat_margins.append(dist - self.cfg.env.threat_safe_dist)
        if uav_margins:
            min_uav_margin = float(min(uav_margins))
            mean_uav_margin = float(np.mean(uav_margins))
        else:
            min_uav_margin = float(self.cfg.env.map_size)
            mean_uav_margin = float(self.cfg.env.map_size)
        if threat_margins:
            min_threat_margin = float(min(threat_margins))
            mean_threat_margin = float(np.mean(threat_margins))
        else:
            min_threat_margin = float(self.cfg.env.map_size)
            mean_threat_margin = float(self.cfg.env.map_size)
        hard_safe = boundary_ok and min_uav_margin > 0.0 and min_threat_margin > 0.0
        return {
            "hard_safe": bool(hard_safe),
            "boundary_ok": bool(boundary_ok),
            "min_uav_uav_margin": float(min_uav_margin),
            "mean_uav_uav_margin": float(mean_uav_margin),
            "min_uav_threat_margin": float(min_threat_margin),
            "mean_uav_threat_margin": float(mean_threat_margin),
        }

    def _default_actions_for_state(self, state: Dict[str, Any]) -> List[int]:
        defaults: List[int] = []
        for i in range(self.cfg.env.n_uavs):
            valid = np.flatnonzero(self._valid_actions_for_state(state, i))
            defaults.append(int(valid[0]))
        return defaults

    def _legacy_enumerate_hard_actions(
        self,
        env: UAVSearchEnv,
        state: Dict[str, Any],
        agent_idx: int,
        base_actions: Sequence[int],
    ) -> List[int]:
        valid_mask = self._valid_actions_for_state(state, agent_idx)
        hard_actions: List[int] = []
        for action in np.flatnonzero(valid_mask):
            candidate_actions = list(base_actions)
            candidate_actions[agent_idx] = int(action)
            next_state = self.predict_next_state(env, candidate_actions, state=state)
            if self.check_hard_constraints(next_state)["hard_safe"]:
                hard_actions.append(int(action))
        return hard_actions

    def _legacy_future_safe_exists(self, env: UAVSearchEnv, state: Dict[str, Any]) -> bool:
        candidate_actions = self._default_actions_for_state(state)
        for agent_idx in range(self.cfg.env.n_uavs):
            hard_actions = self._legacy_enumerate_hard_actions(env, state, agent_idx, candidate_actions)
            if not hard_actions:
                return False
            candidate_actions[agent_idx] = int(hard_actions[0])
        return True

    def _legacy_enumerate_recursive_feasible_actions(
        self,
        env: UAVSearchEnv,
        state: Dict[str, Any],
        agent_idx: int,
        base_actions: Sequence[int],
        hard_actions: Sequence[int],
    ) -> List[int]:
        recursive_actions: List[int] = []
        for action in hard_actions:
            candidate_actions = list(base_actions)
            candidate_actions[agent_idx] = int(action)
            next_state = self.predict_next_state(env, candidate_actions, state=state)
            if self._legacy_future_safe_exists(env, next_state):
                recursive_actions.append(int(action))
        return recursive_actions

    def _candidate_next_positions(self, state: Dict[str, Any], agent_idx: int, action_ids: np.ndarray) -> np.ndarray:
        x, y, z = state["uavs"][agent_idx]
        h_idx = action_ids // len(V_DIRS)
        v_idx = action_ids % len(V_DIRS)
        dxdy = UAV_H_DIRS_ARRAY[h_idx]
        dz = V_DIRS_ARRAY[v_idx]
        next_xyz = np.empty((len(action_ids), 3), dtype=np.int32)
        next_xyz[:, :2] = dxdy + np.asarray([x, y], dtype=np.int32)
        next_xyz[:, 2] = dz + int(z)
        return next_xyz

    def _single_next_position(self, state: Dict[str, Any], agent_idx: int, action: int) -> np.ndarray:
        valid_mask = self._valid_actions_for_state(state, agent_idx)
        chosen = int(action)
        if valid_mask[chosen] == 0:
            chosen = int(np.flatnonzero(valid_mask)[0])
        next_xyz = self._candidate_next_positions(state, agent_idx, np.asarray([chosen], dtype=np.int64))
        next_xyz[:, 0] = np.clip(next_xyz[:, 0], 0, self.cfg.env.map_size - 1)
        next_xyz[:, 1] = np.clip(next_xyz[:, 1], 0, self.cfg.env.map_size - 1)
        next_xyz[:, 2] = np.clip(next_xyz[:, 2], 0, self.cfg.env.n_altitudes - 1)
        return next_xyz[0]

    def _planned_next_positions(
        self,
        state: Dict[str, Any],
        actions: Sequence[int],
        planned_next_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if planned_next_positions is not None:
            return planned_next_positions
        next_positions = np.empty((self.cfg.env.n_uavs, 3), dtype=np.int32)
        for agent_idx, action in enumerate(actions):
            next_positions[agent_idx] = self._single_next_position(state, agent_idx, int(action))
        return next_positions

    def _refine_hard_actions(
        self,
        env: UAVSearchEnv,
        state: Dict[str, Any],
        agent_idx: int,
        base_actions: Sequence[int],
        hard_actions: Sequence[int],
        clearances: Dict[int, float],
    ) -> tuple[List[int], Dict[int, float]]:
        if not self.cfg.shield.refine_enabled or not hard_actions:
            return [int(a) for a in hard_actions], dict(clearances)

        refine_margin = float(self.cfg.shield.refine_margin)
        refine_candidates = [int(a) for a in hard_actions if float(clearances.get(int(a), refine_margin + 1.0)) <= refine_margin]
        if not refine_candidates:
            return [int(a) for a in hard_actions], dict(clearances)

        refine_start = self._maybe_start()
        refined_hard_actions: List[int] = []
        refined_clearances = dict(clearances)
        refine_set = set(refine_candidates)
        for action in hard_actions:
            act = int(action)
            if act not in refine_set:
                refined_hard_actions.append(act)
                continue
            candidate_actions = list(base_actions)
            candidate_actions[agent_idx] = act
            next_state = self.predict_next_state(env, candidate_actions, state=state)
            if self.check_hard_constraints(next_state)["hard_safe"]:
                refined_hard_actions.append(act)
            else:
                refined_clearances.pop(act, None)
        self._maybe_record("refine_time", refine_start)
        return refined_hard_actions, refined_clearances

    def _enumerate_hard_actions_with_meta(
        self,
        env: UAVSearchEnv,
        state: Dict[str, Any],
        agent_idx: int,
        base_actions: Sequence[int],
        planned_next_positions: Optional[np.ndarray] = None,
        allow_refine: bool = True,
    ) -> tuple[List[int], Dict[str, Any]]:
        if self.use_legacy_path_for_profile:
            start = self._maybe_start()
            hard_actions = self._legacy_enumerate_hard_actions(env, state, agent_idx, base_actions)
            hard_meta = {
                "clearances": {},
                "valid_action_count": int(len(hard_actions)),
                "local_uav_count": 0,
                "local_threat_count": 0,
                "near_boundary": False,
                "crowded": False,
                "min_candidate_clearance": None,
                "base_margins": self.check_hard_constraints(state),
            }
            self._maybe_record("hard_time", start)
            return hard_actions, hard_meta
        return self._rule_based_hard_safe_actions(
            env,
            state,
            agent_idx,
            base_actions,
            planned_next_positions=planned_next_positions,
            allow_refine=allow_refine,
        )

    def _rule_based_hard_safe_actions(
        self,
        env: UAVSearchEnv,
        state: Dict[str, Any],
        agent_idx: int,
        base_actions: Sequence[int],
        planned_next_positions: Optional[np.ndarray] = None,
        allow_refine: bool = True,
    ) -> tuple[List[int], Dict[str, Any]]:
        start = self._maybe_start()
        cache_key = (self._state_key(state), int(agent_idx), tuple(int(a) for a in base_actions), bool(allow_refine))
        if self.cache_enabled:
            self.profile["hard_cache_queries"] += 1.0
            cached = self.hard_cache.get(cache_key)
            if cached is not None:
                self.profile["hard_cache_hits"] += 1.0
                self._maybe_record("hard_time", start)
                return list(cached[0]), dict(cached[1])

        rule_start = self._maybe_start()
        valid_mask = self._valid_actions_for_state(state, agent_idx).astype(bool)
        action_ids = np.flatnonzero(valid_mask)
        if len(action_ids) == 0:
            empty_meta = {
                "clearances": {},
                "valid_action_count": 0,
                "local_uav_count": 0,
                "local_threat_count": 0,
                "near_boundary": True,
                "crowded": True,
                "min_candidate_clearance": None,
                "base_margins": self.check_hard_constraints(state),
            }
            if self.cache_enabled:
                self.hard_cache[cache_key] = ([], dict(empty_meta))
            self._maybe_record("rule_mask_time", rule_start)
            self._maybe_record("hard_time", start)
            return [], empty_meta

        candidates = self._candidate_next_positions(state, agent_idx, action_ids)
        candidates[:, 0] = np.clip(candidates[:, 0], 0, self.cfg.env.map_size - 1)
        candidates[:, 1] = np.clip(candidates[:, 1], 0, self.cfg.env.map_size - 1)
        candidates[:, 2] = np.clip(candidates[:, 2], 0, self.cfg.env.n_altitudes - 1)
        curr_xy = np.asarray(state["uavs"][agent_idx][:2], dtype=np.float32)
        base_margins = self.check_hard_constraints(state)

        threats = np.asarray(state["threats"], dtype=np.float32)
        threat_radius = float(self.cfg.env.threat_safe_dist + self.cfg.shield.threat_radius_inflation)
        threat_clearance = np.full(len(action_ids), float(self.cfg.env.map_size), dtype=np.float32)
        threat_violation = np.zeros(len(action_ids), dtype=bool)
        local_threat_count = 0
        if threats.size > 0:
            threat_local_radius = float(self.cfg.env.threat_safe_dist + MAX_HORIZ_STEP + self.cfg.shield.local_threat_padding)
            nearby_threats = np.sum((threats - curr_xy) ** 2, axis=1) <= threat_local_radius ** 2
            local_threats = threats[nearby_threats]
            local_threat_count = int(local_threats.shape[0])
            if local_threat_count > 0:
                diff = candidates[:, None, :2].astype(np.float32) - local_threats[None, :, :]
                d2 = np.sum(diff * diff, axis=-1)
                threat_violation = (d2 <= threat_radius ** 2 + 1e-6).any(axis=1)
                threat_clearance = np.sqrt(d2.min(axis=1)) - threat_radius

        curr_uavs = np.asarray(state["uavs"], dtype=np.int32)
        other_idx = np.array([i for i in range(self.cfg.env.n_uavs) if i != agent_idx], dtype=np.int32)
        local_uav_count = 0
        uav_violation = np.zeros(len(action_ids), dtype=bool)
        swap_violation = np.zeros(len(action_ids), dtype=bool)
        uav_clearance = np.full(len(action_ids), float(self.cfg.env.map_size), dtype=np.float32)
        if other_idx.size > 0:
            base_next_uavs = self._planned_next_positions(state, base_actions, planned_next_positions=planned_next_positions)
            other_curr = curr_uavs[other_idx, :2].astype(np.float32)
            other_next = base_next_uavs[other_idx, :2].astype(np.float32)
            uav_local_radius = float(self.cfg.env.uav_safe_dist + 2.0 * MAX_HORIZ_STEP + self.cfg.shield.local_uav_padding)
            local_curr = np.sum((other_curr - curr_xy) ** 2, axis=1) <= uav_local_radius ** 2
            local_next = np.sum((other_next - curr_xy) ** 2, axis=1) <= uav_local_radius ** 2
            nearby_uavs = np.logical_or(local_curr, local_next)
            local_other_curr = other_curr[nearby_uavs]
            local_other_next = other_next[nearby_uavs]
            local_uav_count = int(local_other_next.shape[0])
            if local_uav_count > 0:
                diff = candidates[:, None, :2].astype(np.float32) - local_other_next[None, :, :]
                d2 = np.sum(diff * diff, axis=-1)
                uav_violation = (d2 <= float(self.cfg.env.uav_safe_dist ** 2) + 1e-6).any(axis=1)
                uav_clearance = np.sqrt(d2.min(axis=1)) - float(self.cfg.env.uav_safe_dist)
                swap_in = np.all(candidates[:, None, :2].astype(np.float32) == local_other_curr[None, :, :], axis=-1)
                swap_out = np.all(local_other_next[None, :, :] == curr_xy[None, None, :], axis=-1)
                swap_violation = np.logical_and(swap_in, swap_out).any(axis=1)

        forbidden = np.logical_or.reduce((threat_violation, uav_violation, swap_violation))
        hard_actions = action_ids[~forbidden].astype(np.int64).tolist()
        clearances = {}
        combined_clearance = np.minimum(uav_clearance, threat_clearance)
        if hard_actions:
            for action, clearance in zip(action_ids.tolist(), combined_clearance.tolist()):
                if action in hard_actions:
                    clearances[int(action)] = float(clearance)
            if allow_refine:
                hard_actions, clearances = self._refine_hard_actions(
                    env,
                    state,
                    agent_idx,
                    base_actions,
                    hard_actions,
                    clearances,
                )

        x, y, z = state["uavs"][agent_idx]
        boundary_margin = min(
            x,
            y,
            self.cfg.env.map_size - 1 - x,
            self.cfg.env.map_size - 1 - y,
            z,
            self.cfg.env.n_altitudes - 1 - z,
        )
        hard_meta = {
            "clearances": clearances,
            "valid_action_count": int(len(action_ids)),
            "local_uav_count": int(local_uav_count),
            "local_threat_count": int(local_threat_count),
            "near_boundary": bool(boundary_margin <= 1),
            "crowded": bool(local_uav_count >= 2),
            "min_candidate_clearance": float(min(clearances.values())) if clearances else float(self.cfg.env.map_size),
            "max_candidate_clearance": float(max(clearances.values())) if clearances else float(self.cfg.env.map_size),
            "base_margins": base_margins,
        }

        if self.cache_enabled:
            self.hard_cache[cache_key] = (list(hard_actions), dict(hard_meta))
        self._maybe_record("rule_mask_time", rule_start)
        self._maybe_record("hard_time", start)
        return hard_actions, hard_meta

    def enumerate_hard_actions(
        self,
        env: UAVSearchEnv,
        state: Dict[str, Any],
        agent_idx: int,
        base_actions: Sequence[int],
        allow_refine: bool = True,
    ) -> List[int]:
        hard_actions, _ = self._enumerate_hard_actions_with_meta(
            env,
            state,
            agent_idx,
            base_actions,
            allow_refine=allow_refine,
        )
        return hard_actions

    def enumerate_safe_actions(
        self,
        env: UAVSearchEnv,
        state: Dict[str, Any],
        agent_idx: int,
        base_actions: Sequence[int],
        allow_refine: bool = True,
    ) -> List[int]:
        # Backward-compatible alias: the old one-step "safe" set in the code
        # base is now explicitly treated as the always-on A_hard layer.
        return self.enumerate_hard_actions(
            env,
            state,
            agent_idx,
            base_actions,
            allow_refine=allow_refine,
        )

    def _uses_legacy_recursive_gate(self) -> bool:
        return bool(self.cfg.shield.legacy_recursive_gate or not self.risk_score_enabled)

    def _zero_agent_risk(self) -> Dict[str, float]:
        return {
            "score": 0.0,
            "clear": 0.0,
            "clear_gap": 0.0,
            "fragility": 0.0,
            "region": 0.0,
            "hist": 0.0,
            "support": 0.0,
            "high_risk": False,
        }

    def _clearance_risk_from_value(self, clearance: float | None, *, norm: float | None = None) -> float:
        if clearance is None:
            return 1.0
        value = float(clearance)
        if not np.isfinite(value):
            return 1.0
        scale = max(float(self.cfg.shield.risk_clearance_norm if norm is None else norm), 1e-6)
        return float(np.clip(1.0 - value / scale, 0.0, 1.0))

    def _clearance_gap_risk_from_value(self, clearance_gap: float | None) -> float:
        if clearance_gap is None:
            return 1.0
        gap = float(clearance_gap)
        if not np.isfinite(gap):
            return 1.0
        norm = max(float(self.cfg.shield.risk_clear_gap_norm), 1e-6)
        return float(np.clip(gap / norm, 0.0, 1.0))

    def compute_clear_risk(self, hard_actions: Sequence[int], hard_meta: Dict[str, Any]) -> float:
        # The v1 risk is post-A_hard / pre-A_rec: clearance is reused from the
        # already-computed A_hard geometry statistics instead of triggering a
        # new expensive search.
        if not hard_actions:
            return 1.0
        min_candidate_clearance = hard_meta.get("min_candidate_clearance")
        return self._clearance_risk_from_value(min_candidate_clearance)

    def compute_proposed_clear_risk(
        self,
        proposed_action: int,
        hard_actions: Sequence[int],
        hard_meta: Dict[str, Any],
    ) -> float:
        # risk_base aligns the primary clear term with the actor's current proposal.
        # If the proposal is already outside A_hard, A_hard will intervene anyway,
        # so this upgrade-specific term stays inactive and other cheap terms can
        # still decide whether A_rec is worth the extra cost.
        if not hard_actions:
            return 1.0
        if int(proposed_action) not in {int(a) for a in hard_actions}:
            return 0.0
        clearances = hard_meta.get("clearances", {})
        return self._clearance_risk_from_value(clearances.get(int(proposed_action)))

    def compute_clear_gap_risk(
        self,
        proposed_action: int,
        hard_actions: Sequence[int],
        hard_meta: Dict[str, Any],
    ) -> float:
        if not hard_actions:
            return 1.0
        if int(proposed_action) not in {int(a) for a in hard_actions}:
            return 0.0
        clearances = hard_meta.get("clearances", {})
        proposed_clearance = clearances.get(int(proposed_action))
        if proposed_clearance is None or not np.isfinite(float(proposed_clearance)):
            return 1.0
        best_clearance = hard_meta.get("max_candidate_clearance")
        if best_clearance is None:
            finite_values = [float(v) for v in clearances.values() if np.isfinite(float(v))]
            best_clearance = max(finite_values) if finite_values else None
        if best_clearance is None:
            return 1.0
        gap = max(0.0, float(best_clearance) - float(proposed_clearance))
        return self._clearance_gap_risk_from_value(gap)

    def compute_support_risk(self, hard_actions: Sequence[int], hard_meta: Dict[str, Any]) -> float:
        if not hard_actions:
            return 1.0
        clearances = hard_meta.get("clearances", {})
        if not clearances:
            return 1.0
        robust_margin = float(self.cfg.shield.risk_support_clearance_margin)
        robust_count = 0
        for action in hard_actions:
            clearance = clearances.get(int(action))
            if clearance is not None and np.isfinite(float(clearance)) and float(clearance) >= robust_margin:
                robust_count += 1
        return float(1.0 - robust_count / max(len(hard_actions), 1))

    def compute_fragility_risk(self, hard_actions: Sequence[int], hard_meta: Dict[str, Any]) -> float:
        valid_action_count = int(hard_meta.get("valid_action_count", len(hard_actions)))
        if valid_action_count <= 0:
            return 1.0
        return float(np.clip(1.0 - len(hard_actions) / max(valid_action_count, 1), 0.0, 1.0))

    def compute_region_risk(self, hard_meta: Dict[str, Any]) -> float:
        boundary = 1.0 if bool(hard_meta.get("near_boundary", False)) else 0.0
        threat_norm = max(float(self.cfg.shield.risk_threat_count_norm), 1e-6)
        local_threat_count = max(0.0, float(hard_meta.get("local_threat_count", 0)))
        threat_risk = float(np.clip(local_threat_count / threat_norm, 0.0, 1.0))
        crowded = 1.0 if bool(hard_meta.get("crowded", False)) else 0.0
        return float((boundary + threat_risk + crowded) / 3.0)

    def compute_hist_risk(self, agent_idx: int) -> float:
        window = max(1, int(self.cfg.shield.risk_hist_window))
        if agent_idx < 0 or agent_idx >= len(self.agent_intervention_history):
            return 0.0
        history = self.agent_intervention_history[agent_idx]
        return float(sum(int(v) for v in history) / window)

    def compute_agent_risk(
        self,
        agent_idx: int,
        hard_actions: Sequence[int],
        hard_meta: Dict[str, Any],
        proposed_action: int | None = None,
    ) -> Dict[str, float]:
        if not self.risk_score_enabled:
            return self._zero_agent_risk()

        risk_variant = canonicalize_risk_variant(str(getattr(self.cfg.shield, "risk_variant", "v1")))
        region_risk = self.compute_region_risk(hard_meta)
        hist_risk = 0.0
        clear_risk = 0.0
        clear_gap_risk = 0.0
        fragility_risk = 0.0
        support_risk = 0.0

        if risk_variant == "risk_base":
            chosen_action = -1 if proposed_action is None else int(proposed_action)
            clear_risk = self.compute_proposed_clear_risk(chosen_action, hard_actions, hard_meta)
            clear_gap_risk = self.compute_clear_gap_risk(chosen_action, hard_actions, hard_meta)
            support_risk = self.compute_support_risk(hard_actions, hard_meta)
            score = (
                float(self.cfg.shield.risk_base_weight_prop_clear) * clear_risk
                + float(self.cfg.shield.risk_base_weight_clear_gap) * clear_gap_risk
                + float(self.cfg.shield.risk_base_weight_support) * support_risk
                + float(self.cfg.shield.risk_base_weight_region) * region_risk
            )
        elif risk_variant == "v_next2":
            # v_next2 keeps the proposed-action clearance view, but replaces the
            # proposed-vs-best gap term with A_hard fragility because the
            # eligible-only offline validation showed clear_gap hurting ranking.
            chosen_action = -1 if proposed_action is None else int(proposed_action)
            clear_risk = self.compute_proposed_clear_risk(chosen_action, hard_actions, hard_meta)
            fragility_risk = self.compute_fragility_risk(hard_actions, hard_meta)
            support_risk = self.compute_support_risk(hard_actions, hard_meta)
            score = (
                float(self.cfg.shield.risk_vnext2_weight_prop_clear) * clear_risk
                + float(self.cfg.shield.risk_vnext2_weight_fragility) * fragility_risk
                + float(self.cfg.shield.risk_vnext2_weight_support) * support_risk
                + float(self.cfg.shield.risk_vnext2_weight_region) * region_risk
            )
        else:
            clear_risk = self.compute_clear_risk(hard_actions, hard_meta)
            hist_risk = self.compute_hist_risk(agent_idx)
            score = (
                float(self.cfg.shield.risk_weight_clear) * clear_risk
                + float(self.cfg.shield.risk_weight_region) * region_risk
                + float(self.cfg.shield.risk_weight_hist) * hist_risk
            )
        # TODO: extend with feasibility-proxy and uncertainty / preference-conflict
        # terms while keeping the risk path cheap and fully reusable from shield
        # local geometry / cache statistics.
        return {
            "score": float(score),
            "clear": float(clear_risk),
            "clear_gap": float(clear_gap_risk),
            "fragility": float(fragility_risk),
            "region": float(region_risk),
            "hist": float(hist_risk),
            "support": float(support_risk),
            "high_risk": bool(score >= float(self.cfg.shield.risk_threshold)),
        }

    def _future_safe_exists(self, env: UAVSearchEnv, state: Dict[str, Any]) -> bool:
        start = self._maybe_start()
        state_key = self._state_key(state)
        if self.cache_enabled:
            self.profile["future_cache_queries"] += 1.0
            cached = self.future_safe_cache.get(state_key)
            if cached is not None:
                self.profile["future_cache_hits"] += 1.0
                self._maybe_record("recursive_time", start)
                return bool(cached)

        candidate_actions = self._default_actions_for_state(state)
        future_safe = True
        for agent_idx in range(self.cfg.env.n_uavs):
            hard_actions = self.enumerate_hard_actions(
                env,
                state,
                agent_idx,
                candidate_actions,
                allow_refine=False,
            )
            if not hard_actions:
                future_safe = False
                break
            candidate_actions[agent_idx] = int(hard_actions[0])

        if self.cache_enabled:
            self.future_safe_cache[state_key] = bool(future_safe)
        self._maybe_record("recursive_time", start)
        return bool(future_safe)

    def enumerate_recursive_feasible_actions(
        self,
        env: UAVSearchEnv,
        state: Dict[str, Any],
        agent_idx: int,
        base_actions: Sequence[int],
        hard_actions: Sequence[int],
    ) -> List[int]:
        if self.use_legacy_path_for_profile:
            return self._legacy_enumerate_recursive_feasible_actions(env, state, agent_idx, base_actions, hard_actions)

        start = self._maybe_start()
        recursive_actions: List[int] = []
        for action in hard_actions:
            candidate_actions = list(base_actions)
            candidate_actions[agent_idx] = int(action)
            next_state = self.predict_next_state(env, candidate_actions, state=state)
            if self._future_safe_exists(env, next_state):
                recursive_actions.append(int(action))
        self._maybe_record("recursive_time", start)
        return recursive_actions

    def _current_step_near_risk_zone(self, state: Dict[str, Any], agent_idx: int, hard_meta: Dict[str, Any]) -> bool:
        if hard_meta.get("near_boundary", False):
            return True
        if hard_meta.get("local_threat_count", 0) > 0:
            return True
        if hard_meta.get("crowded", False):
            return True
        x, y, _ = state["uavs"][agent_idx]
        return x <= 1 or y <= 1 or x >= self.cfg.env.map_size - 2 or y >= self.cfg.env.map_size - 2

    def _should_run_recursive_check_legacy(
        self,
        state: Dict[str, Any],
        agent_idx: int,
        proposed_action: int,
        hard_actions: Sequence[int],
        hard_meta: Dict[str, Any],
    ) -> bool:
        if self.cfg.shield.mode != "recursive":
            return False
        if proposed_action not in hard_actions:
            return True
        # Historical config name kept for compatibility: this threshold is
        # applied to the always-on A_hard set size.
        if len(hard_actions) <= int(self.cfg.shield.recursive_safe_action_threshold):
            return True
        min_candidate_clearance = float(hard_meta.get("min_candidate_clearance", self.cfg.env.map_size))
        if min_candidate_clearance < float(self.cfg.shield.recursive_margin_threshold):
            return True
        risk_zone = bool(
            hard_meta.get("near_boundary", False)
            or hard_meta.get("local_threat_count", 0) > 0
            or hard_meta.get("crowded", False)
        )
        if risk_zone and min_candidate_clearance < float(self.cfg.shield.recursive_margin_threshold) + 0.25:
            return True
        if (
            sum(self.recent_trigger_history) >= int(self.cfg.shield.recursive_recent_trigger_threshold)
            and min_candidate_clearance < float(self.cfg.shield.recursive_margin_threshold) + 0.25
        ):
            return True
        return False

    def _should_run_recursive_check(
        self,
        state: Dict[str, Any],
        agent_idx: int,
        proposed_action: int,
        hard_actions: Sequence[int],
        hard_meta: Dict[str, Any],
        risk_info: Dict[str, float],
    ) -> bool:
        if self.cfg.shield.mode != "recursive":
            return False
        if self._uses_legacy_recursive_gate():
            return self._should_run_recursive_check_legacy(
                state,
                agent_idx,
                proposed_action,
                hard_actions,
                hard_meta,
            )
        return bool(risk_info.get("high_risk", False))

    def _actor_scores(self, actor_output: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(actor_output, torch.Tensor):
            return actor_output.detach().float().cpu().numpy()
        return np.asarray(actor_output, dtype=np.float32)

    def _prioritized_recursive_candidates(
        self,
        actor_output: torch.Tensor | np.ndarray,
        hard_actions: Sequence[int],
        proposed_action: int,
        clearances: Dict[int, float],
    ) -> List[int]:
        hard_set = {int(a) for a in hard_actions}
        ordered: List[int] = []
        seen = set()

        def add(action: int) -> None:
            act = int(action)
            if act in hard_set and act not in seen:
                ordered.append(act)
                seen.add(act)

        add(proposed_action)
        scores = self._actor_scores(actor_output)
        if hard_actions:
            sorted_hard = sorted((int(a) for a in hard_actions), key=lambda a: float(scores[a]), reverse=True)
            for action in sorted_hard[: max(1, int(self.cfg.shield.candidate_top_k))]:
                add(action)
        if clearances:
            best_clearance = max(clearances.items(), key=lambda kv: float(kv[1]))[0]
            add(int(best_clearance))
        return ordered

    def _decision_recursive_actions(
        self,
        env: UAVSearchEnv,
        state: Dict[str, Any],
        agent_idx: int,
        base_actions: Sequence[int],
        hard_actions: Sequence[int],
        proposed_action: int,
        actor_output: torch.Tensor | np.ndarray,
        hard_meta: Dict[str, Any],
        risk_info: Dict[str, float],
    ) -> tuple[List[int], Dict[str, Any]]:
        decision_meta = {
            "recursive_gate_run": False,
            "high_risk": bool(risk_info.get("high_risk", False)),
            "used_legacy_recursive_gate": bool(self._uses_legacy_recursive_gate()),
        }
        if not hard_actions:
            return [], decision_meta
        if self.use_legacy_path_for_profile:
            decision_meta["recursive_gate_run"] = True
            self.profile["recursive_gate_runs"] += 1.0
            self.profile["recursive_candidate_checks"] += float(len(hard_actions))
            return (
                self._legacy_enumerate_recursive_feasible_actions(
                    env,
                    state,
                    agent_idx,
                    base_actions,
                    hard_actions,
                ),
                decision_meta,
            )
        if not self._should_run_recursive_check(
            state,
            agent_idx,
            proposed_action,
            hard_actions,
            hard_meta,
            risk_info,
        ):
            self.profile["recursive_gate_skips"] += 1.0
            return list(hard_actions), decision_meta

        decision_meta["recursive_gate_run"] = True
        self.profile["recursive_gate_runs"] += 1.0
        prioritized = self._prioritized_recursive_candidates(
            actor_output,
            hard_actions,
            proposed_action,
            hard_meta.get("clearances", {}),
        )
        rec_actions: List[int] = []
        checked = set()

        def evaluate_candidates(candidates: Sequence[int]) -> None:
            for action in candidates:
                act = int(action)
                if act in checked:
                    continue
                checked.add(act)
                self.profile["recursive_candidate_checks"] += 1.0
                candidate_actions = list(base_actions)
                candidate_actions[agent_idx] = act
                next_state = self.predict_next_state(env, candidate_actions, state=state)
                if self._future_safe_exists(env, next_state):
                    rec_actions.append(act)

        evaluate_candidates(prioritized)
        if rec_actions:
            return rec_actions, decision_meta

        if self.cfg.shield.candidate_full_fallback:
            remaining = [int(a) for a in hard_actions if int(a) not in checked]
            evaluate_candidates(remaining)
        return rec_actions, decision_meta

    def mask_actor_output_with_allowed_set(
        self,
        actor_output: torch.Tensor | np.ndarray,
        allowed_actions: Sequence[int],
        valid_mask: Sequence[bool] | np.ndarray,
    ) -> tuple[torch.Tensor | np.ndarray, np.ndarray]:
        allowed_mask = np.zeros(len(valid_mask), dtype=bool)
        allowed_mask[np.asarray(allowed_actions, dtype=np.int64)] = True
        allowed_mask &= np.asarray(valid_mask, dtype=bool)
        if isinstance(actor_output, torch.Tensor):
            out = actor_output.clone()
            out[~torch.as_tensor(allowed_mask, dtype=torch.bool, device=actor_output.device)] = -1e9
            return out, allowed_mask
        out = np.array(actor_output, copy=True)
        out[~allowed_mask] = -1e9
        return out, allowed_mask

    def mask_actor_output_with_safe_set(
        self,
        actor_output: torch.Tensor | np.ndarray,
        safe_actions: Sequence[int],
        valid_mask: Sequence[bool] | np.ndarray,
    ) -> tuple[torch.Tensor | np.ndarray, np.ndarray]:
        # Backward-compatible alias for older helper naming.
        return self.mask_actor_output_with_allowed_set(actor_output, safe_actions, valid_mask)

    def resample_action_from_allowed_set(
        self,
        actor_output: torch.Tensor | np.ndarray,
        allowed_actions: Sequence[int],
        valid_mask: Sequence[bool] | np.ndarray,
        selection_mode: str,
    ) -> tuple[int, np.ndarray]:
        masked_output, allowed_mask = self.mask_actor_output_with_allowed_set(actor_output, allowed_actions, valid_mask)
        if not allowed_mask.any():
            raise ValueError("allowed action set is empty")
        if selection_mode == "sample":
            if not isinstance(masked_output, torch.Tensor):
                logits = torch.tensor(masked_output, dtype=torch.float32)
                action = int(Categorical(logits=logits).sample().item())
            else:
                action = int(Categorical(logits=masked_output).sample().item())
            return action, allowed_mask
        if isinstance(masked_output, torch.Tensor):
            return int(masked_output.argmax(dim=-1).item()), allowed_mask
        return int(np.argmax(masked_output)), allowed_mask

    def resample_action_from_safe_set(
        self,
        actor_output: torch.Tensor | np.ndarray,
        safe_actions: Sequence[int],
        valid_mask: Sequence[bool] | np.ndarray,
        selection_mode: str,
    ) -> tuple[int, np.ndarray]:
        # Backward-compatible alias for older helper naming.
        return self.resample_action_from_allowed_set(
            actor_output,
            safe_actions,
            valid_mask,
            selection_mode,
        )

    def apply(
        self,
        env: UAVSearchEnv,
        proposed_actions: Sequence[int],
        actor_preferences: torch.Tensor | np.ndarray,
        action_masks: Sequence[Sequence[bool]] | np.ndarray,
        selection_mode: str,
    ) -> tuple[List[int], np.ndarray, Dict[str, Any]]:
        total_start = self._maybe_start()
        proposed = [int(a) for a in proposed_actions]
        valid_masks = np.asarray(action_masks, dtype=bool)
        state = self._capture_state(env)
        self.profile["steps"] += 1.0

        if not self.enabled:
            final_state = self.predict_next_state(env, proposed, state=state)
            margins = self.check_hard_constraints(final_state)
            near_miss = (
                margins["min_uav_uav_margin"] < float(self.cfg.shield.near_miss_margin)
                or margins["min_uav_threat_margin"] < float(self.cfg.shield.near_miss_margin)
            )
            # Keep legacy *_safe_* keys as aliases for the A_hard statistics so
            # older analysis scripts keep working unchanged.
            step_stats = {
                "shield_triggered": False,
                "shield_trigger_count": int(self.shield_trigger_count),
                "shield_agent_trigger_count": int(self.shield_agent_trigger_count),
                "shield_triggered_agents": 0,
                "action_replaced": False,
                "action_replaced_agents": 0,
                "shield_action_replaced_count": int(self.action_replaced_count),
                "shield_a_hard_size": 0.0,
                "shield_a_safe_size": 0.0,
                "shield_a_hard_sizes": [0 for _ in proposed],
                "shield_a_rec_size": 0.0,
                "shield_a_safe_sizes": [0 for _ in proposed],
                "shield_a_rec_sizes": [0 for _ in proposed],
                "hard_action_count": 0.0,
                "safe_action_count": 0.0,
                "rec_action_count": 0.0,
                "min_hard_action_count_step": 0.0,
                "min_safe_action_count_step": 0.0,
                "min_rec_action_count_step": 0.0,
                "dead_end_hard": False,
                "dead_end_safe": False,
                "dead_end_rec": False,
                "shield_fallback_triggered": False,
                "shield_fallback_count": int(self.fallback_count),
                "min_uav_uav_margin": float(margins["min_uav_uav_margin"]),
                "mean_uav_uav_margin": float(margins["mean_uav_uav_margin"]),
                "min_uav_threat_margin": float(margins["min_uav_threat_margin"]),
                "mean_uav_threat_margin": float(margins["mean_uav_threat_margin"]),
                "near_miss": bool(near_miss),
                "shield_penalty": 0.0,
                "risk_score": 0.0,
                "risk_clear": 0.0,
                "risk_clear_gap": 0.0,
                "risk_fragility": 0.0,
                "risk_region": 0.0,
                "risk_hist": 0.0,
                "risk_support": 0.0,
                "high_risk_agents": 0,
                "high_risk_rate_step": 0.0,
                "recursive_gate_agents": 0,
                "recursive_gate_rate_step": 0.0,
                "risk_agent_count": int(self.cfg.env.n_uavs),
            }
            self._maybe_record("shield_time", total_start)
            return proposed, valid_masks.copy(), step_stats

        final_actions = list(proposed)
        effective_masks = valid_masks.copy()
        planned_next_positions = self._planned_next_positions(state, final_actions)
        hard_sizes: List[int] = []
        rec_sizes: List[int] = []
        shield_triggered = False
        action_replaced = False
        replaced_agents = 0
        triggered_agents = 0
        fallback_triggered = False
        recursive_gate_ran = False
        risk_scores: List[float] = []
        risk_clear_scores: List[float] = []
        risk_clear_gap_scores: List[float] = []
        risk_fragility_scores: List[float] = []
        risk_region_scores: List[float] = []
        risk_hist_scores: List[float] = []
        risk_support_scores: List[float] = []
        high_risk_agents = 0
        recursive_gate_agents = 0

        # Shield hierarchy: A_hard -> risk gate -> A_rec.
        # "safe" mode is intentionally kept as the experiment name for the
        # hard-safe-only mode, i.e. it computes A_hard and never upgrades.
        for agent_idx in range(self.cfg.env.n_uavs):
            base_actions = list(final_actions)
            gate_runs_before = self.profile["recursive_gate_runs"]
            hard_actions, hard_meta = self._enumerate_hard_actions_with_meta(
                env,
                state,
                agent_idx,
                base_actions,
                planned_next_positions=planned_next_positions,
            )
            hard_sizes.append(len(hard_actions))
            risk_info = self.compute_agent_risk(agent_idx, hard_actions, hard_meta, proposed_action=final_actions[agent_idx])
            risk_scores.append(float(risk_info["score"]))
            risk_clear_scores.append(float(risk_info["clear"]))
            risk_clear_gap_scores.append(float(risk_info.get("clear_gap", 0.0)))
            risk_fragility_scores.append(float(risk_info.get("fragility", 0.0)))
            risk_region_scores.append(float(risk_info["region"]))
            risk_hist_scores.append(float(risk_info["hist"]))
            risk_support_scores.append(float(risk_info.get("support", 0.0)))
            high_risk_agents += int(bool(risk_info.get("high_risk", False)))

            rec_actions: List[int] = []
            if self.cfg.shield.mode == "recursive":
                rec_actions, decision_meta = self._decision_recursive_actions(
                    env,
                    state,
                    agent_idx,
                    base_actions,
                    hard_actions,
                    final_actions[agent_idx],
                    actor_preferences[agent_idx],
                    hard_meta,
                    risk_info,
                )
                recursive_gate_agents += int(bool(decision_meta.get("recursive_gate_run", False)))
            else:
                decision_meta = {
                    "recursive_gate_run": False,
                    "high_risk": bool(risk_info.get("high_risk", False)),
                    "used_legacy_recursive_gate": bool(self._uses_legacy_recursive_gate()),
                }
            recursive_gate_ran = recursive_gate_ran or (self.profile["recursive_gate_runs"] > gate_runs_before)
            if self.cfg.shield.mode == "recursive":
                rec_sizes.append(len(rec_actions))

            candidate_actions = hard_actions
            if self.cfg.shield.mode == "recursive":
                candidate_actions = rec_actions if rec_actions else hard_actions

            proposed_action = final_actions[agent_idx]
            proposed_is_admissible = proposed_action in candidate_actions if candidate_actions else False
            if self.cfg.shield.mode == "safe":
                proposed_is_admissible = proposed_action in hard_actions

            agent_intervened = False
            if proposed_is_admissible:
                self.agent_intervention_history[agent_idx].append(0)
                continue

            shield_triggered = True
            triggered_agents += 1
            if candidate_actions:
                selected_action, selected_mask = self.resample_action_from_allowed_set(
                    actor_preferences[agent_idx],
                    candidate_actions,
                    valid_masks[agent_idx],
                    selection_mode=selection_mode,
                )
                effective_masks[agent_idx] = selected_mask
            else:
                fallback_triggered = True
                self.fallback_count += 1
                fallback_candidates = np.flatnonzero(valid_masks[agent_idx]).tolist()
                if fallback_candidates:
                    selected_action, selected_mask = self.resample_action_from_allowed_set(
                        actor_preferences[agent_idx],
                        fallback_candidates,
                        valid_masks[agent_idx],
                        selection_mode=selection_mode,
                    )
                    effective_masks[agent_idx] = selected_mask
                else:
                    selected_action = proposed_action
            if selected_action != proposed_action:
                agent_intervened = True
                action_replaced = True
                replaced_agents += 1
                self.action_replaced_count += 1
            final_actions[agent_idx] = int(selected_action)
            planned_next_positions[agent_idx] = self._single_next_position(state, agent_idx, int(selected_action))
            self.agent_intervention_history[agent_idx].append(1 if agent_intervened else 0)

        if shield_triggered:
            self.shield_trigger_count += 1
            self.shield_agent_trigger_count += triggered_agents
        self.recent_trigger_history.append(1 if shield_triggered else 0)
        if self.cfg.shield.mode == "recursive":
            if recursive_gate_ran:
                self.profile["recursive_gate_step_runs"] += 1.0
            else:
                self.profile["recursive_gate_step_skips"] += 1.0

        final_state = self.predict_next_state(env, final_actions, state=state)
        margins = self.check_hard_constraints(final_state)
        near_miss = (
            margins["min_uav_uav_margin"] < float(self.cfg.shield.near_miss_margin)
            or margins["min_uav_threat_margin"] < float(self.cfg.shield.near_miss_margin)
        )
        penalty = float(self.cfg.shield.penalty_coef) if shield_triggered else 0.0
        # Keep legacy *_safe_* keys as aliases for the A_hard statistics so
        # older analysis scripts keep working unchanged.
        step_stats = {
            "shield_triggered": bool(shield_triggered),
            "shield_trigger_count": int(self.shield_trigger_count),
            "shield_agent_trigger_count": int(self.shield_agent_trigger_count),
            "shield_triggered_agents": int(triggered_agents),
            "action_replaced": bool(action_replaced),
            "action_replaced_agents": int(replaced_agents),
            "shield_action_replaced_count": int(self.action_replaced_count),
            "shield_a_hard_size": float(np.mean(hard_sizes)) if hard_sizes else 0.0,
            "shield_a_safe_size": float(np.mean(hard_sizes)) if hard_sizes else 0.0,
            "shield_a_rec_size": float(np.mean(rec_sizes)) if rec_sizes else 0.0,
            "shield_a_hard_sizes": list(hard_sizes),
            "shield_a_safe_sizes": list(hard_sizes),
            "shield_a_rec_sizes": list(rec_sizes),
            "hard_action_count": float(np.mean(hard_sizes)) if hard_sizes else 0.0,
            "safe_action_count": float(np.mean(hard_sizes)) if hard_sizes else 0.0,
            "rec_action_count": float(np.mean(rec_sizes)) if rec_sizes else 0.0,
            "min_hard_action_count_step": float(np.min(hard_sizes)) if hard_sizes else 0.0,
            "min_safe_action_count_step": float(np.min(hard_sizes)) if hard_sizes else 0.0,
            "min_rec_action_count_step": float(np.min(rec_sizes)) if rec_sizes else 0.0,
            "dead_end_hard": bool(any(size == 0 for size in hard_sizes)),
            "dead_end_safe": bool(any(size == 0 for size in hard_sizes)),
            "dead_end_rec": bool(any(size == 0 for size in rec_sizes)),
            "shield_fallback_triggered": bool(fallback_triggered),
            "shield_fallback_count": int(self.fallback_count),
            "min_uav_uav_margin": float(margins["min_uav_uav_margin"]),
            "mean_uav_uav_margin": float(margins["mean_uav_uav_margin"]),
            "min_uav_threat_margin": float(margins["min_uav_threat_margin"]),
            "mean_uav_threat_margin": float(margins["mean_uav_threat_margin"]),
            "near_miss": bool(near_miss),
            "shield_penalty": penalty,
            "recursive_gate_run": bool(recursive_gate_ran),
            "risk_score": float(np.mean(risk_scores)) if risk_scores else 0.0,
            "risk_clear": float(np.mean(risk_clear_scores)) if risk_clear_scores else 0.0,
            "risk_clear_gap": float(np.mean(risk_clear_gap_scores)) if risk_clear_gap_scores else 0.0,
            "risk_fragility": float(np.mean(risk_fragility_scores)) if risk_fragility_scores else 0.0,
            "risk_region": float(np.mean(risk_region_scores)) if risk_region_scores else 0.0,
            "risk_hist": float(np.mean(risk_hist_scores)) if risk_hist_scores else 0.0,
            "risk_support": float(np.mean(risk_support_scores)) if risk_support_scores else 0.0,
            "high_risk_agents": int(high_risk_agents),
            "high_risk_rate_step": float(high_risk_agents / max(1, self.cfg.env.n_uavs)),
            "recursive_gate_agents": int(recursive_gate_agents),
            "recursive_gate_rate_step": float(recursive_gate_agents / max(1, self.cfg.env.n_uavs)),
            "risk_agent_count": int(self.cfg.env.n_uavs),
        }
        self._maybe_record("shield_time", total_start)
        return final_actions, effective_masks, step_stats

    def apply_reward_penalty(self, reward: float, info: Dict[str, Any], step_stats: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        merged_info = dict(info)
        merged_info.update(step_stats)
        penalty = float(step_stats.get("shield_penalty", 0.0))
        original_reward = float(reward)
        penalized_reward = original_reward - penalty
        merged_info["original_reward"] = original_reward
        merged_info["shield_penalized_reward"] = penalized_reward
        if "sparse_reward" in merged_info:
            merged_info["original_sparse_reward"] = float(merged_info["sparse_reward"])
            merged_info["shield_penalized_sparse_reward"] = float(merged_info["sparse_reward"]) - penalty
        return penalized_reward, merged_info
