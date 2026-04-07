from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EnvConfig, RewardConfig
from .maps import CognitiveMaps


UAV_H_DIRS = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]
THREAT_H_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
V_DIRS = [-1, 0, 1]


@dataclass
class TargetState:
    x: float
    y: float
    theta: float


class UAVSearchEnv:
    def __init__(self, env_cfg: EnvConfig, rew_cfg: RewardConfig, seed: int = 42):
        self.cfg = env_cfg
        self.rew_cfg = rew_cfg
        self.rng = np.random.default_rng(seed)
        self.n_h_actions = len(UAV_H_DIRS)
        self.n_actions = self.n_h_actions * len(V_DIRS)
        self.reset()

    def reset(self):
        s = self.cfg.map_size
        self.t = 0
        self.uavs = []
        for _ in range(self.cfg.n_uavs):
            self.uavs.append([self.rng.integers(0, s), self.rng.integers(0, s), self.rng.integers(0, self.cfg.n_altitudes)])
        self.prev_dirs: List[Optional[Tuple[int, int]]] = [None] * self.cfg.n_uavs
        self.prev_actions = [None] * self.cfg.n_uavs

        self.targets: List[TargetState] = []
        for _ in range(self.cfg.n_targets):
            self.targets.append(
                TargetState(
                    x=float(self.rng.uniform(0, s - 1)),
                    y=float(self.rng.uniform(0, s - 1)),
                    theta=float(self.rng.uniform(0, 2 * math.pi)),
                )
            )

        self.threats = [[self.rng.integers(0, s), self.rng.integers(0, s)] for _ in range(self.cfg.n_threats)]
        self.maps = [CognitiveMaps(s) for _ in range(self.cfg.n_uavs)]

        self.coverage = np.zeros((s, s), dtype=np.int8)
        self.found_targets = set()
        self.step_visible_target_ids = set()
        self.step_positive_target_ids = set()
        self.collisions = 0
        self.total_reward = 0.0
        self.last_entropy = self._global_entropy()
        self.step_new_coverage = 0
        self.step_turns = [False] * self.cfg.n_uavs
        return self._build_obs()

    def _global_entropy(self):
        tpm = np.mean([m.tpm for m in self.maps], axis=0)
        p = np.clip(tpm, 1e-6, 1 - 1e-6)
        return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).sum())

    def _global_tpm(self):
        return np.mean([m.tpm for m in self.maps], axis=0)

    def global_state(self) -> np.ndarray:
        s = self.cfg.map_size
        occ = np.zeros((s, s), dtype=np.float32)
        uav_scale = 1.0 / float(max(1, self.cfg.n_uavs))
        for x, y, _ in self.uavs:
            occ[x, y] += uav_scale
        for tx, ty in self.threats:
            occ[tx, ty] = -1.0

        mean_dpm = np.mean([m.dpm for m in self.maps], axis=0)
        mean_stm = np.mean([m.stm.astype(np.float32) for m in self.maps], axis=0)
        stm_scale = float(max(1, self.cfg.max_steps, self.t))
        stm_norm = mean_stm / stm_scale

        return np.stack(
            [
                self._global_tpm().astype(np.float32),
                mean_dpm.astype(np.float32),
                stm_norm.astype(np.float32),
                self.coverage.astype(np.float32),
                occ,
            ],
            axis=0,
        )

    def _target_grid(self):
        s = self.cfg.map_size
        grid = np.zeros((s, s), dtype=np.int8)
        for j, tar in enumerate(self.targets):
            tx, ty = int(round(tar.x)), int(round(tar.y))
            tx = int(np.clip(tx, 0, s - 1))
            ty = int(np.clip(ty, 0, s - 1))
            grid[tx, ty] = 1
        return grid

    def _target_cell_to_id(self):
        s = self.cfg.map_size
        out = {}
        for tid, tar in enumerate(self.targets):
            tx, ty = int(round(tar.x)), int(round(tar.y))
            tx = int(np.clip(tx, 0, s - 1))
            ty = int(np.clip(ty, 0, s - 1))
            out[(tx, ty)] = tid
        return out

    def _move_targets(self):
        s = self.cfg.map_size
        occ = set()
        for t in self.targets:
            noise = self.rng.normal(0, 0.4)
            t.theta = self.cfg.target_alpha * t.theta + (1 - self.cfg.target_alpha) * (t.theta + noise)
            t.x += self.cfg.target_speed * math.cos(t.theta)
            t.y += self.cfg.target_speed * math.sin(t.theta)
            t.x = float(np.clip(t.x, 0, s - 1))
            t.y = float(np.clip(t.y, 0, s - 1))
            cell = (int(round(t.x)), int(round(t.y)))
            while cell in occ:
                t.theta += self.rng.uniform(-0.5, 0.5)
                t.x = float(np.clip(t.x + 0.2 * math.cos(t.theta), 0, s - 1))
                t.y = float(np.clip(t.y + 0.2 * math.sin(t.theta), 0, s - 1))
                cell = (int(round(t.x)), int(round(t.y)))
            occ.add(cell)

    def _move_threats(self):
        if not self.cfg.dynamic_threat or self.t % self.cfg.threat_move_period != 0:
            return
        s = self.cfg.map_size
        for tr in self.threats:
            cand = []
            for dx, dy in THREAT_H_DIRS:
                nx, ny = tr[0] + dx, tr[1] + dy
                if 0 <= nx < s and 0 <= ny < s:
                    cand.append((nx, ny))
            if cand:
                tr[0], tr[1] = cand[self.rng.integers(0, len(cand))]

    def valid_actions(self, i: int) -> np.ndarray:
        s = self.cfg.map_size
        x, y, z = self.uavs[i]
        prev = self.prev_dirs[i]
        mask = np.zeros(self.n_actions, dtype=np.int8)
        fallback_angles = np.full(self.n_actions, np.inf, dtype=np.float32)
        for a in range(self.n_actions):
            h, v = divmod(a, len(V_DIRS))
            dx, dy = UAV_H_DIRS[h]
            nz = z + V_DIRS[v]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < s and 0 <= ny < s):
                continue
            if not (0 <= nz < self.cfg.n_altitudes):
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
            if ang <= self.cfg.max_turn_rad + 1e-6:
                mask[a] = 1
        if mask.sum() == 0:
            best = float(np.min(fallback_angles))
            if np.isfinite(best):
                mask[fallback_angles <= best + 1e-6] = 1
        return mask

    def _neighbors(self, idx: int) -> List[int]:
        out = []
        xi, yi, _ = self.uavs[idx]
        for j, (xj, yj, _) in enumerate(self.uavs):
            if j == idx:
                continue
            if (xi - xj) ** 2 + (yi - yj) ** 2 <= self.cfg.comm_radius ** 2:
                out.append(j)
        return out

    def _observe_cells(self, x: int, y: int, r: int):
        cells = []
        s = self.cfg.map_size
        for i in range(max(0, x - r), min(s, x + r + 1)):
            for j in range(max(0, y - r), min(s, y + r + 1)):
                if (i - x) ** 2 + (j - y) ** 2 <= r * r:
                    cells.append((i, j))
        return cells

    def step(self, actions: List[int]):
        s = self.cfg.map_size
        self.t += 1
        prev_dirs = list(self.prev_dirs)
        self.step_turns = [False] * self.cfg.n_uavs

        for i, a in enumerate(actions):
            mask = self.valid_actions(i)
            if mask[a] == 0:
                a = int(np.flatnonzero(mask)[0])
            h, v = divmod(a, len(V_DIRS))
            self.prev_actions[i] = a
            dx, dy = UAV_H_DIRS[h]
            self.step_turns[i] = prev_dirs[i] is not None and prev_dirs[i] != (dx, dy)
            self.prev_dirs[i] = (dx, dy)
            self.uavs[i][0] = int(np.clip(self.uavs[i][0] + dx, 0, s - 1))
            self.uavs[i][1] = int(np.clip(self.uavs[i][1] + dy, 0, s - 1))
            self.uavs[i][2] = int(np.clip(self.uavs[i][2] + V_DIRS[v], 0, self.cfg.n_altitudes - 1))

        self._move_targets()
        self._move_threats()
        target_grid = self._target_grid()
        target_cell_to_id = self._target_cell_to_id()
        self.step_visible_target_ids = set()
        self.step_positive_target_ids = set()

        self.step_new_coverage = 0
        for i, (x, y, z) in enumerate(self.uavs):
            m = self.maps[i]
            r = self.cfg.sense_radii[z]
            pd = self.cfg.pd_levels[z]
            pf = self.cfg.pf_levels[z]
            cells = self._observe_cells(x, y, r)
            detections = []
            for (cx, cy) in cells:
                has_target = target_grid[cx, cy] == 1
                p = pd if has_target else pf
                detection = 1 if self.rng.random() < p else 0
                detections.append(detection)
                if has_target:
                    tid = target_cell_to_id.get((cx, cy))
                    if tid is not None:
                        self.step_visible_target_ids.add(tid)
                if has_target and detection == 1:
                    tid = target_cell_to_id.get((cx, cy))
                    if tid is not None:
                        self.step_positive_target_ids.add(tid)
                m.stm[cx, cy] = self.t
                if self.coverage[cx, cy] == 0:
                    self.step_new_coverage += 1
                self.coverage[cx, cy] = 1
            m.stage1_detection_update(cells, detections, pd=pd, pf=pf)
            neighbor_maps = [self.maps[j] for j in self._neighbors(i)]
            m.stage2_fusion_update(neighbor_maps)
            if self.rew_cfg.use_compensation:
                m.stage3_revisit_compensation(self.t, self.cfg.t0, self.cfg.p_delta)
            revisit_mask = (self.t - m.stm) > self.cfg.t0
            m.update_dpm(
                self.t,
                zeta_p=self.cfg.zeta_p,
                xi_p=self.cfg.xi_p,
                ea=self.cfg.ea,
                da=self.cfg.da,
                ga=self.cfg.ga,
                revisit_mask=revisit_mask,
            )

        reward, info = self._calc_reward(target_grid)
        self.total_reward += reward
        all_targets_found = len(self.found_targets) == self.cfg.n_targets
        done = self.t >= self.cfg.max_steps or (
            self.cfg.terminate_on_all_targets_found and all_targets_found
        )
        obs = self._build_obs()
        return obs, reward, done, info

    def _calc_reward(self, target_grid: np.ndarray):
        # r1 discovery:
        # strict mode requires a positive detection in the current timestep;
        # non-strict mode only requires the target to enter any UAV's FOV.
        global_tpm = self._global_tpm()
        new_found = 0
        for tid, tar in enumerate(self.targets):
            tx, ty = int(round(tar.x)), int(round(tar.y))
            tx = int(np.clip(tx, 0, self.cfg.map_size - 1))
            ty = int(np.clip(ty, 0, self.cfg.map_size - 1))
            if tid in self.found_targets:
                continue
            if self.cfg.strict_found_detection:
                if tid in self.step_positive_target_ids:
                    self.found_targets.add(tid)
                    new_found += 1
            elif tid in self.step_visible_target_ids:
                self.found_targets.add(tid)
                new_found += 1

        reward = 0.0
        dense = self.rew_cfg.mode != "sparse"
        sparse = self.rew_cfg.mode == "sparse"
        discovery_reward = 0.0

        if new_found:
            discovery_reward = new_found * max(0.0, self.rew_cfg.r1_discovery - self.t)
            reward += discovery_reward

        # r3 collision penalty: the paper distinguishes UAV-UAV and UAV-threat collisions.
        uav_collisions = 0
        threat_collisions = 0
        for i in range(len(self.uavs)):
            xi, yi, _ = self.uavs[i]
            for j in range(i + 1, len(self.uavs)):
                xj, yj, _ = self.uavs[j]
                if (xi - xj) ** 2 + (yi - yj) ** 2 <= self.cfg.uav_safe_dist ** 2:
                    uav_collisions += 1
            for tx, ty in self.threats:
                if (xi - tx) ** 2 + (yi - ty) ** 2 <= self.cfg.threat_safe_dist ** 2:
                    threat_collisions += 1
        col = uav_collisions + threat_collisions
        self.collisions += col
        collision_reward = self.rew_cfg.r2_time * uav_collisions + self.rew_cfg.r3_collision * threat_collisions
        reward += collision_reward
        sparse_reward = discovery_reward + collision_reward

        if dense:
            # r2 time penalty
            reward += self.rew_cfg.r2_time

            # r4 uncertainty reduction reward
            ent = self._global_entropy()
            reward += self.rew_cfg.r4_entropy * (self.last_entropy - ent)
            self.last_entropy = ent

            # r5 revisit pheromone reward follows the paper's map-wise sum by
            # default; an optional normalization keeps its scale comparable to
            # the other reward terms in engineering experiments.
            dpm_gain = float(sum(m.dpm.sum() for m in self.maps))
            dpm_reward = self.rew_cfg.r5_pheromone * dpm_gain
            if self.rew_cfg.normalize_dpm_reward:
                dpm_reward /= float(max(1, self.cfg.n_uavs * self.cfg.map_size * self.cfg.map_size))

            # r7 energy penalty: penalize turns unless the UAV is already locked onto a target.
            energy_turn_penalty = 0
            for i, (x, y, _) in enumerate(self.uavs):
                if self.rew_cfg.use_energy_penalty and self.step_turns[i] and self.maps[i].tpm[x, y] <= self.cfg.xi_p:
                    energy_turn_penalty += 1
            reward += dpm_reward

            # r6 coverage reward uses the cumulative visited-area ratio.
            reward += self.rew_cfg.r6_coverage * float(self.coverage.mean())
            reward -= self.rew_cfg.r7_energy_penalty * (energy_turn_penalty / float(max(1, self.cfg.n_uavs)))

        info = {
            "search_rate": len(self.found_targets) / self.cfg.n_targets,
            "found_targets": float(len(self.found_targets)),
            "coverage_rate": float(self.coverage.mean()),
            "coverage_ratio": float(self.coverage.mean()),
            "collisions": col,
            "uav_collisions": uav_collisions,
            "threat_collisions": threat_collisions,
            "error_rate": float(np.abs(global_tpm - target_grid).mean()),
            "new_found": new_found,
            "discovery_reward": float(discovery_reward),
            "collision_reward": float(collision_reward),
            "dpm_reward": float(dpm_reward) if dense else 0.0,
            "sparse_reward": float(sparse_reward),
            "dense_mode": dense and not sparse,
        }
        return reward, info

    def _patch(self, arr: np.ndarray, x: int, y: int):
        p = self.cfg.patch_size
        r = p // 2
        out = np.zeros((p, p), dtype=np.float32)
        s = self.cfg.map_size
        for i in range(p):
            for j in range(p):
                gx, gy = x - r + i, y - r + j
                if 0 <= gx < s and 0 <= gy < s:
                    out[i, j] = arr[gx, gy]
        return out

    def _build_obs(self):
        obs = []
        s = self.cfg.map_size
        for i, (x, y, z) in enumerate(self.uavs):
            m = self.maps[i]
            occ = np.zeros((s, s), dtype=np.float32)
            for j, (ux, uy, _) in enumerate(self.uavs):
                if i != j:
                    occ[ux, uy] = 1.0
            for tx, ty in self.threats:
                occ[tx, ty] = -1.0
            chans = [
                self._patch(m.tpm, x, y),
                self._patch(m.dpm, x, y),
                self._patch(m.stm.astype(np.float32), x, y),
                self._patch(occ, x, y),
            ]
            obs_tensor = np.stack(chans, axis=0)
            prev_action_feat = np.zeros(self.n_actions, dtype=np.float32)
            if self.prev_actions[i] is not None:
                prev_action_feat[int(self.prev_actions[i])] = 1.0
            extra = np.concatenate([
                np.eye(self.cfg.n_altitudes, dtype=np.float32)[z],
                prev_action_feat,
            ])
            obs.append({"map": obs_tensor, "extra": extra, "action_mask": self.valid_actions(i)})
        return obs

    def clone_for_reward(self, mode: str):
        self.rew_cfg.mode = mode
