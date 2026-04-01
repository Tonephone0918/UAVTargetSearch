from __future__ import annotations

import numpy as np


class CognitiveMaps:
    """STM, TPM, DPM, and detection counters with 3-stage TPM update."""

    def __init__(self, map_size: int):
        self.map_size = map_size
        self.stm = np.zeros((map_size, map_size), dtype=np.int32)
        self.tpm = np.full((map_size, map_size), 0.5, dtype=np.float32)
        self.dpm = np.zeros((map_size, map_size), dtype=np.float32)
        self.n_plus = np.zeros((map_size, map_size), dtype=np.int32)
        self.n_minus = np.zeros((map_size, map_size), dtype=np.int32)

    @staticmethod
    def _to_log_odds(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _from_log_odds(l: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-l))

    def stage1_detection_update(
        self,
        observed_cells: list[tuple[int, int]],
        detections: list[int],
        pd: float,
        pf: float,
    ) -> None:
        eps = 1e-6
        lr_pos = np.log((pd + eps) / (pf + eps))
        lr_neg = np.log((1 - pd + eps) / (1 - pf + eps))
        for (x, y), z in zip(observed_cells, detections):
            l = self._to_log_odds(np.array([self.tpm[x, y]], dtype=np.float32))[0]
            l = l + (lr_pos if z == 1 else lr_neg)
            self.tpm[x, y] = float(self._from_log_odds(np.array([l]))[0])
            if z == 1:
                self.n_plus[x, y] += 1
            else:
                self.n_minus[x, y] += 1

    def stage2_fusion_update(self, neighbor_maps: list["CognitiveMaps"]) -> None:
        if not neighbor_maps:
            return
        nplus_stack = [self.n_plus] + [n.n_plus for n in neighbor_maps]
        nminus_stack = [self.n_minus] + [n.n_minus for n in neighbor_maps]
        fused_plus = np.maximum.reduce(nplus_stack)
        fused_minus = np.maximum.reduce(nminus_stack)
        self.n_plus = fused_plus
        self.n_minus = fused_minus

        # Rebuild TPM from fused counts with Beta(1,1) style smoothing
        self.tpm = (1.0 + fused_plus) / (2.0 + fused_plus + fused_minus)

    def stage3_revisit_compensation(self, timestep: int, t0: int, p_delta: float) -> None:
        stale = (timestep - self.stm) > t0
        if not stale.any():
            return

        # Paper-aligned revisit compensation:
        # stale cells only increase toward 0.5 and never exceed it.
        stale_vals = self.tpm[stale]
        raised = np.where(
            stale_vals < (0.5 - p_delta),
            stale_vals + p_delta,
            0.5,
        )
        self.tpm[stale] = np.clip(raised, 0.0, 0.5)
        self.n_plus[stale] = 0
        self.n_minus[stale] = 0

    def update_dpm(
        self,
        timestep: int,
        zeta_p: float,
        xi_p: float,
        ea: float,
        da: float,
        ga: float,
        revisit_mask: np.ndarray,
    ) -> None:
        self.dpm *= (1.0 - ea)

        release_mask = ((self.tpm >= zeta_p) & (self.tpm < xi_p)) | revisit_mask
        self.dpm[release_mask] += da

        if ga > 0:
            d = self.dpm
            padded = np.pad(d, 1)
            neigh_avg = (
                padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
            ) / 4.0
            self.dpm = (1 - ga) * self.dpm + ga * neigh_avg

        self.dpm[self.tpm >= xi_p] = 0.0
