from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import EnvConfig, RewardConfig
from .maps import CognitiveMaps


H_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
        self.n_actions = 12
        self.reset()

    def reset(self):
        s = self.cfg.map_size
        self.t = 0
        self.uavs = []
        for _ in range(self.cfg.n_uavs):
            self.uavs.append([self.rng.integers(0, s), self.rng.integers(0, s), self.rng.integers(0, self.cfg.n_altitudes)])
        self.prev_dirs = [(1, 0)] * self.cfg.n_uavs

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

    def _target_grid(self):
        s = self.cfg.map_size
        grid = np.zeros((s, s), dtype=np.int8)
        for j, tar in enumerate(self.targets):
            tx, ty = int(round(tar.x)), int(round(tar.y))
            tx = int(np.clip(tx, 0, s - 1))
            ty = int(np.clip(ty, 0, s - 1))
            grid[tx, ty] = 1
        return grid

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
            for dx, dy in H_DIRS:
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
        for a in range(self.n_actions):
            h = a // 3
            v = a % 3
            dx, dy = H_DIRS[h]
            nz = z + V_DIRS[v]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < s and 0 <= ny < s):
                continue
            if not (0 <= nz < self.cfg.n_altitudes):
                continue
            dot = prev[0] * dx + prev[1] * dy
            prev_norm = math.sqrt(prev[0] ** 2 + prev[1] ** 2)
            ang = math.acos(np.clip(dot / (prev_norm + 1e-6), -1.0, 1.0))
            if ang > self.cfg.max_turn_rad:
                continue
            mask[a] = 1
        if mask.sum() == 0:
            mask[:] = 1
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
            h, v = a // 3, a % 3
            dx, dy = H_DIRS[h]
            self.step_turns[i] = (prev_dirs[i] != (dx, dy))
            self.prev_dirs[i] = (dx, dy)
            self.uavs[i][0] = int(np.clip(self.uavs[i][0] + dx, 0, s - 1))
            self.uavs[i][1] = int(np.clip(self.uavs[i][1] + dy, 0, s - 1))
            self.uavs[i][2] = int(np.clip(self.uavs[i][2] + V_DIRS[v], 0, self.cfg.n_altitudes - 1))

        self._move_targets()
        self._move_threats()
        target_grid = self._target_grid()

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
                detections.append(1 if self.rng.random() < p else 0)
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
        done = len(self.found_targets) == self.cfg.n_targets or self.t >= self.cfg.max_steps
        obs = self._build_obs()
        return obs, reward, done, info

    def _calc_reward(self, target_grid: np.ndarray):
        # r1 discovery
        new_found = 0
        for tid, tar in enumerate(self.targets):
            tx, ty = int(round(tar.x)), int(round(tar.y))
            if tid in self.found_targets:
                continue
            if any((ux == tx and uy == ty) for ux, uy, _ in self.uavs):
                self.found_targets.add(tid)
                new_found += 1

        reward = 0.0
        dense = self.rew_cfg.mode != "sparse"
        sparse = self.rew_cfg.mode == "sparse"
        discovery_reward = 0.0

        if new_found:
            discovery_reward = new_found * max(0.0, self.rew_cfg.r1_discovery - self.t)
            reward += discovery_reward

        # collisions
        col = 0
        for i in range(len(self.uavs)):
            xi, yi, _ = self.uavs[i]
            for j in range(i + 1, len(self.uavs)):
                xj, yj, _ = self.uavs[j]
                if (xi - xj) ** 2 + (yi - yj) ** 2 <= self.cfg.uav_safe_dist ** 2:
                    col += 1
            for tx, ty in self.threats:
                if (xi - tx) ** 2 + (yi - ty) ** 2 <= self.cfg.threat_safe_dist ** 2:
                    col += 1
        self.collisions += col
        collision_reward = 0.0
        if col > 0:
            collision_reward = self.rew_cfg.r3_collision * col
            reward += collision_reward
        sparse_reward = discovery_reward + collision_reward

        if dense:
            reward += self.rew_cfg.r2_time
            ent = self._global_entropy()
            reward += self.rew_cfg.r4_entropy * max(0.0, self.last_entropy - ent)
            self.last_entropy = ent

            dpm_gain = 0.0
            energy_turn_penalty = 0
            for i, (x, y, _) in enumerate(self.uavs):
                dpm_gain += self.maps[i].dpm[x, y]
                if self.rew_cfg.use_energy_penalty and self.step_turns[i] and self.maps[i].tpm[x, y] <= self.cfg.xi_p:
                    energy_turn_penalty += 1
            reward += self.rew_cfg.r5_pheromone * dpm_gain
            reward += self.rew_cfg.r6_coverage * (self.step_new_coverage / float(self.cfg.map_size * self.cfg.map_size))
            reward -= self.rew_cfg.r7_energy_penalty * (energy_turn_penalty / float(max(1, self.cfg.n_uavs)))

        info = {
            "search_rate": len(self.found_targets) / self.cfg.n_targets,
            "coverage_rate": float(self.coverage.mean()),
            "collisions": col,
            "error_rate": float(np.abs(np.mean([m.tpm for m in self.maps], axis=0) - target_grid).mean()),
            "new_found": new_found,
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
            extra = np.concatenate([
                np.eye(self.cfg.n_altitudes, dtype=np.float32)[z],
                np.eye(4, dtype=np.float32)[H_DIRS.index(self.prev_dirs[i]) if self.prev_dirs[i] in H_DIRS else 0],
                np.eye(self.cfg.n_uavs, dtype=np.float32)[i],
            ])
            obs.append({"map": obs_tensor, "extra": extra, "action_mask": self.valid_actions(i)})
        return obs

    def clone_for_reward(self, mode: str):
        self.rew_cfg.mode = mode
