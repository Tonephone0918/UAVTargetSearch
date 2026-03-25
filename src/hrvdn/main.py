from __future__ import annotations

import argparse

from .config import ExperimentConfig
from .trainer import HRVDNTrainer


def run_ablation(kind: str, device: str):
    cfg = ExperimentConfig()
    if kind == "height":
        for n, pd, pf, sr in [
            (1, [0.82], [0.18], [2]),
            (3, [0.94, 0.82, 0.70], [0.05, 0.18, 0.30], [1, 2, 3]),
            (5, [0.97, 0.90, 0.82, 0.76, 0.70], [0.03, 0.08, 0.18, 0.24, 0.30], [1, 1, 2, 2, 3]),
        ]:
            cfg.env.n_altitudes = n
            cfg.env.pd_levels = pd
            cfg.env.pf_levels = pf
            cfg.env.sense_radii = sr
            trainer = HRVDNTrainer(cfg, device=device)
            hist = trainer.train()
            print("height", n, hist[-1] if hist else None)
    elif kind == "compensation":
        for flag in [True, False]:
            cfg.reward.use_compensation = flag
            trainer = HRVDNTrainer(cfg, device=device)
            hist = trainer.train()
            print("compensation", flag, hist[-1] if hist else None)
    elif kind == "energy":
        for flag in [True, False]:
            cfg.reward.use_energy_penalty = flag
            trainer = HRVDNTrainer(cfg, device=device)
            hist = trainer.train()
            print("energy", flag, hist[-1] if hist else None)
    elif kind == "reward":
        for mode in ["dense", "sparse", "hybrid"]:
            cfg.reward.mode = mode
            trainer = HRVDNTrainer(cfg, device=device)
            hist = trainer.train()
            print("reward", mode, hist[-1] if hist else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", choices=["height", "compensation", "energy", "reward"], default=None)
    p.add_argument("--dense-epochs", type=int, default=None)
    p.add_argument("--sparse-epochs", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint, e.g. checkpoints/latest.pt")
    args = p.parse_args()

    if args.ablation:
        run_ablation(args.ablation, args.device)
        return

    cfg = ExperimentConfig()
    if args.dense_epochs is not None:
        cfg.train.dense_epochs = args.dense_epochs
    if args.sparse_epochs is not None:
        cfg.train.sparse_epochs = args.sparse_epochs

    trainer = HRVDNTrainer(cfg, device=args.device)
    trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()
