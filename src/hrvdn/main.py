from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig, canonicalize_risk_variant
from .mappo_trainer import MAPPOTrainer
from .rollout_vis import generate_baseline_rollout_html, generate_rollout_html
from .runtime import apply_env_overrides
from .trainer import HRVDNTrainer
from .validate import evaluate_checkpoint, format_metrics
from .visualize import generate_training_report, parse_tags


def run_ablation(kind: str, device: str, cfg: ExperimentConfig, algo: str):
    trainer_cls = MAPPOTrainer if algo == "mappo" else HRVDNTrainer
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
            trainer = trainer_cls(cfg, device=device)
            hist = trainer.train()
            print("height", n, hist[-1] if hist else None)
    elif kind == "compensation":
        for flag in [True, False]:
            cfg.reward.use_compensation = flag
            trainer = trainer_cls(cfg, device=device)
            hist = trainer.train()
            print("compensation", flag, hist[-1] if hist else None)
    elif kind == "energy":
        for flag in [True, False]:
            cfg.reward.use_energy_penalty = flag
            trainer = trainer_cls(cfg, device=device)
            hist = trainer.train()
            print("energy", flag, hist[-1] if hist else None)
    elif kind == "reward":
        for mode in ["dense", "sparse", "hybrid"]:
            cfg.reward.mode = mode
            trainer = trainer_cls(cfg, device=device)
            hist = trainer.train()
            print("reward", mode, hist[-1] if hist else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["hrvdn", "mappo"], default="mappo")
    p.add_argument("--ablation", choices=["height", "compensation", "energy", "reward"], default=None)
    p.add_argument("--dense-epochs", type=int, default=2000)
    p.add_argument("--sparse-epochs", type=int, default=0)
    p.add_argument(
        "--normalize-dpm-reward",
        action="store_true",
        help="Normalize the DPM reward by n_uavs * map_size * map_size.",
    )
    p.add_argument("--map-size", type=int, default=None)
    p.add_argument("--n-uavs", type=int, default=None)
    p.add_argument("--n-targets", type=int, default=None)
    p.add_argument("--n-threats", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--terminate-on-all-found",
        action="store_true",
        help="End an episode immediately when all targets have been found.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--shield-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the centralized safety shield before env.step().",
    )
    p.add_argument(
        "--shield-mode",
        choices=["off", "safe", "recursive"],
        default=None,
        help="Shield mode: off, hard-safe-only ('safe' mode), or recursive A_rec upgrade on top of A_hard.",
    )
    p.add_argument(
        "--shield-log-stats",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log shield statistics to stdout / TensorBoard when available.",
    )
    p.add_argument(
        "--shield-profile-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable lightweight shield timing/profile statistics.",
    )
    p.add_argument(
        "--shield-cache-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable lightweight caches for A_hard-set and future-safe queries.",
    )
    p.add_argument(
        "--shield-refine-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable exact gray-zone refinement after cheap rule-based A_hard masking.",
    )
    p.add_argument(
        "--shield-refine-margin",
        type=float,
        default=None,
        help="Only gray-zone candidates with clearance <= margin are exactly re-checked.",
    )
    p.add_argument("--shield-penalty-coef", type=float, default=None, help="Penalty coefficient for shield intervention.")
    p.add_argument(
        "--shield-near-miss-margin",
        type=float,
        default=None,
        help="Near-miss margin threshold used only for safety statistics.",
    )
    p.add_argument(
        "--shield-risk-score-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the configured continuous risk score used to gate A_hard -> A_rec.",
    )
    p.add_argument(
        "--shield-risk-variant",
        choices=["v1", "risk_base", "v_next", "v_next2"],
        default=None,
        help="Risk variant: baseline v1 clear+region+hist, risk_base prop_clear+clear_gap+support+region, or v_next2 prop_clear+fragility+support+region. Legacy alias v_next is still accepted.",
    )
    p.add_argument(
        "--shield-risk-weight-clear",
        type=float,
        default=None,
        help="Weight for the clearance-risk term.",
    )
    p.add_argument(
        "--shield-risk-weight-region",
        type=float,
        default=None,
        help="Weight for the region-risk term.",
    )
    p.add_argument(
        "--shield-risk-weight-hist",
        type=float,
        default=None,
        help="Weight for the history-risk term.",
    )
    p.add_argument(
        "--shield-risk-clearance-norm",
        type=float,
        default=None,
        help="Normalization constant M_c for clearance risk.",
    )
    p.add_argument(
        "--shield-risk-clear-gap-norm",
        type=float,
        default=None,
        help="Normalization constant for the risk_base proposed-vs-best clearance gap risk.",
    )
    p.add_argument(
        "--shield-risk-support-clearance-margin",
        type=float,
        default=None,
        help="Minimum clearance treated as robust support inside A_hard for the risk_base support-risk term.",
    )
    p.add_argument(
        "--shield-risk-base-weight-prop-clear",
        "--shield-risk-vnext-weight-prop-clear",
        dest="shield_risk_base_weight_prop_clear",
        type=float,
        default=None,
        help="Weight for the risk_base proposed-action clearance risk term.",
    )
    p.add_argument(
        "--shield-risk-base-weight-clear-gap",
        "--shield-risk-vnext-weight-clear-gap",
        dest="shield_risk_base_weight_clear_gap",
        type=float,
        default=None,
        help="Weight for the risk_base proposed-vs-best clearance gap term.",
    )
    p.add_argument(
        "--shield-risk-base-weight-support",
        "--shield-risk-vnext-weight-support",
        dest="shield_risk_base_weight_support",
        type=float,
        default=None,
        help="Weight for the risk_base robust-support risk term.",
    )
    p.add_argument(
        "--shield-risk-base-weight-region",
        "--shield-risk-vnext-weight-region",
        dest="shield_risk_base_weight_region",
        type=float,
        default=None,
        help="Weight for the risk_base region-risk term.",
    )
    p.add_argument(
        "--shield-risk-vnext2-weight-prop-clear",
        type=float,
        default=None,
        help="Weight for the v_next2 proposed-action clearance risk term.",
    )
    p.add_argument(
        "--shield-risk-vnext2-weight-fragility",
        type=float,
        default=None,
        help="Weight for the v_next2 hard-set fragility term.",
    )
    p.add_argument(
        "--shield-risk-vnext2-weight-support",
        type=float,
        default=None,
        help="Weight for the v_next2 robust-support risk term.",
    )
    p.add_argument(
        "--shield-risk-vnext2-weight-region",
        type=float,
        default=None,
        help="Weight for the v_next2 region-risk term.",
    )
    p.add_argument(
        "--shield-risk-hist-window",
        type=int,
        default=None,
        help="Sliding window size W for history risk.",
    )
    p.add_argument(
        "--shield-risk-threshold",
        type=float,
        default=None,
        help="Recursive gate threshold eta for the continuous risk score.",
    )
    p.add_argument(
        "--shield-risk-threat-count-norm",
        type=float,
        default=None,
        help="Normalization constant for local threat-count risk.",
    )
    p.add_argument(
        "--shield-legacy-recursive-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the previous binary high-risk gate instead of the continuous risk score.",
    )
    p.add_argument(
        "--shield-progressive-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reserved hook for later progressive shielding work.",
    )
    p.add_argument(
        "--shield-lookahead-horizon",
        type=int,
        default=None,
        help="Reserved hook for later look-ahead shielding work.",
    )
    p.add_argument(
        "--shield-risk-schedule-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reserved hook for later risk-aware scheduling work.",
    )
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint, e.g. checkpoints/latest.pt")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for saved checkpoints.")
    p.add_argument("--tensorboard-dir", type=str, default=None, help="Directory for TensorBoard logs.")
    p.add_argument("--skip-train", action="store_true", help="Skip training and only run validation/report.")
    p.add_argument("--validate-checkpoint", type=str, default=None, help="Evaluate a checkpoint path.")
    p.add_argument("--eval-episodes", type=int, default=10, help="Episodes for checkpoint validation.")
    p.add_argument("--report", action="store_true", help="Generate HTML visualization report from TensorBoard logs.")
    p.add_argument("--report-logdir", type=str, default=None, help="TensorBoard logdir (default from config).")
    p.add_argument("--report-output", type=str, default="runs/hrvdn/report.html", help="Output HTML report path.")
    p.add_argument(
        "--report-tags",
        type=str,
        default=None,
        help="Comma-separated scalar tags for report, e.g. train/episode_reward,eval/search_rate",
    )
    p.add_argument(
        "--animate-checkpoint",
        type=str,
        default=None,
        help="Generate dynamic search replay HTML from checkpoint.",
    )
    p.add_argument(
        "--animate-baseline",
        choices=["greedy"],
        default=None,
        help="Generate dynamic search replay HTML from a built-in baseline policy.",
    )
    p.add_argument(
        "--animate-output",
        type=str,
        default="runs/hrvdn/search_replay.html",
        help="Output HTML path for dynamic replay.",
    )
    p.add_argument(
        "--animate-max-steps",
        type=int,
        default=None,
        help="Optional max steps for replay generation.",
    )
    args = p.parse_args()

    cfg = ExperimentConfig()
    if args.algo == "mappo":
        if cfg.train.checkpoint_dir == "checkpoints":
            cfg.train.checkpoint_dir = "checkpoints/mappo"
        if cfg.train.tensorboard_dir == "runs/hrvdn":
            cfg.train.tensorboard_dir = "runs/mappo"
    if args.checkpoint_dir is not None:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.tensorboard_dir is not None:
        cfg.train.tensorboard_dir = args.tensorboard_dir
    if args.dense_epochs is not None:
        cfg.train.dense_epochs = args.dense_epochs
    if args.sparse_epochs is not None:
        cfg.train.sparse_epochs = args.sparse_epochs
    if args.normalize_dpm_reward:
        cfg.reward.normalize_dpm_reward = True
    if args.shield_enabled is not None:
        cfg.shield.enabled = args.shield_enabled
    if args.shield_mode is not None:
        cfg.shield.mode = args.shield_mode
        cfg.shield.enabled = args.shield_mode != "off"
    elif cfg.shield.enabled and cfg.shield.mode == "off":
        cfg.shield.mode = "safe"
    elif not cfg.shield.enabled:
        cfg.shield.mode = "off"
    if args.shield_log_stats is not None:
        cfg.shield.log_stats = args.shield_log_stats
    if args.shield_profile_enabled is not None:
        cfg.shield.profile_enabled = args.shield_profile_enabled
    if args.shield_cache_enabled is not None:
        cfg.shield.cache_enabled = args.shield_cache_enabled
    if args.shield_refine_enabled is not None:
        cfg.shield.refine_enabled = args.shield_refine_enabled
    if args.shield_refine_margin is not None:
        cfg.shield.refine_margin = args.shield_refine_margin
    if args.shield_penalty_coef is not None:
        cfg.shield.penalty_coef = args.shield_penalty_coef
    if args.shield_near_miss_margin is not None:
        cfg.shield.near_miss_margin = args.shield_near_miss_margin
    if args.shield_risk_score_enabled is not None:
        cfg.shield.risk_score_enabled = args.shield_risk_score_enabled
    if args.shield_risk_variant is not None:
        cfg.shield.risk_variant = canonicalize_risk_variant(args.shield_risk_variant)
    if args.shield_risk_weight_clear is not None:
        cfg.shield.risk_weight_clear = args.shield_risk_weight_clear
    if args.shield_risk_weight_region is not None:
        cfg.shield.risk_weight_region = args.shield_risk_weight_region
    if args.shield_risk_weight_hist is not None:
        cfg.shield.risk_weight_hist = args.shield_risk_weight_hist
    if args.shield_risk_clearance_norm is not None:
        cfg.shield.risk_clearance_norm = args.shield_risk_clearance_norm
    if args.shield_risk_clear_gap_norm is not None:
        cfg.shield.risk_clear_gap_norm = args.shield_risk_clear_gap_norm
    if args.shield_risk_support_clearance_margin is not None:
        cfg.shield.risk_support_clearance_margin = args.shield_risk_support_clearance_margin
    if args.shield_risk_base_weight_prop_clear is not None:
        cfg.shield.risk_base_weight_prop_clear = args.shield_risk_base_weight_prop_clear
    if args.shield_risk_base_weight_clear_gap is not None:
        cfg.shield.risk_base_weight_clear_gap = args.shield_risk_base_weight_clear_gap
    if args.shield_risk_base_weight_support is not None:
        cfg.shield.risk_base_weight_support = args.shield_risk_base_weight_support
    if args.shield_risk_base_weight_region is not None:
        cfg.shield.risk_base_weight_region = args.shield_risk_base_weight_region
    if args.shield_risk_vnext2_weight_prop_clear is not None:
        cfg.shield.risk_vnext2_weight_prop_clear = args.shield_risk_vnext2_weight_prop_clear
    if args.shield_risk_vnext2_weight_fragility is not None:
        cfg.shield.risk_vnext2_weight_fragility = args.shield_risk_vnext2_weight_fragility
    if args.shield_risk_vnext2_weight_support is not None:
        cfg.shield.risk_vnext2_weight_support = args.shield_risk_vnext2_weight_support
    if args.shield_risk_vnext2_weight_region is not None:
        cfg.shield.risk_vnext2_weight_region = args.shield_risk_vnext2_weight_region
    if args.shield_risk_hist_window is not None:
        cfg.shield.risk_hist_window = args.shield_risk_hist_window
    if args.shield_risk_threshold is not None:
        cfg.shield.risk_threshold = args.shield_risk_threshold
    if args.shield_risk_threat_count_norm is not None:
        cfg.shield.risk_threat_count_norm = args.shield_risk_threat_count_norm
    if args.shield_legacy_recursive_gate is not None:
        cfg.shield.legacy_recursive_gate = args.shield_legacy_recursive_gate
    if args.shield_progressive_enabled is not None:
        cfg.shield.progressive_enabled = args.shield_progressive_enabled
    if args.shield_lookahead_horizon is not None:
        cfg.shield.lookahead_horizon = args.shield_lookahead_horizon
    if args.shield_risk_schedule_enabled is not None:
        cfg.shield.risk_schedule_enabled = args.shield_risk_schedule_enabled
    apply_env_overrides(
        cfg,
        map_size=args.map_size,
        n_uavs=args.n_uavs,
        n_targets=args.n_targets,
        n_threats=args.n_threats,
        max_steps=args.max_steps,
        terminate_on_all_targets_found=True if args.terminate_on_all_found else None,
        seed=args.seed,
    )
    env_override_kwargs = {
        "map_size": args.map_size,
        "n_uavs": args.n_uavs,
        "n_targets": args.n_targets,
        "n_threats": args.n_threats,
        "max_steps": args.max_steps,
        "terminate_on_all_targets_found": True if args.terminate_on_all_found else None,
        "seed": args.seed,
    }
    shield_override_kwargs = {}
    if args.shield_enabled is not None or args.shield_mode is not None:
        shield_override_kwargs["enabled"] = cfg.shield.enabled
        shield_override_kwargs["mode"] = cfg.shield.mode
    if args.shield_log_stats is not None:
        shield_override_kwargs["log_stats"] = cfg.shield.log_stats
    if args.shield_profile_enabled is not None:
        shield_override_kwargs["profile_enabled"] = cfg.shield.profile_enabled
    if args.shield_cache_enabled is not None:
        shield_override_kwargs["cache_enabled"] = cfg.shield.cache_enabled
    if args.shield_refine_enabled is not None:
        shield_override_kwargs["refine_enabled"] = cfg.shield.refine_enabled
    if args.shield_refine_margin is not None:
        shield_override_kwargs["refine_margin"] = cfg.shield.refine_margin
    if args.shield_penalty_coef is not None:
        shield_override_kwargs["penalty_coef"] = cfg.shield.penalty_coef
    if args.shield_near_miss_margin is not None:
        shield_override_kwargs["near_miss_margin"] = cfg.shield.near_miss_margin
    if args.shield_risk_score_enabled is not None:
        shield_override_kwargs["risk_score_enabled"] = cfg.shield.risk_score_enabled
    if args.shield_risk_variant is not None:
        shield_override_kwargs["risk_variant"] = cfg.shield.risk_variant
    if args.shield_risk_weight_clear is not None:
        shield_override_kwargs["risk_weight_clear"] = cfg.shield.risk_weight_clear
    if args.shield_risk_weight_region is not None:
        shield_override_kwargs["risk_weight_region"] = cfg.shield.risk_weight_region
    if args.shield_risk_weight_hist is not None:
        shield_override_kwargs["risk_weight_hist"] = cfg.shield.risk_weight_hist
    if args.shield_risk_clearance_norm is not None:
        shield_override_kwargs["risk_clearance_norm"] = cfg.shield.risk_clearance_norm
    if args.shield_risk_clear_gap_norm is not None:
        shield_override_kwargs["risk_clear_gap_norm"] = cfg.shield.risk_clear_gap_norm
    if args.shield_risk_support_clearance_margin is not None:
        shield_override_kwargs["risk_support_clearance_margin"] = cfg.shield.risk_support_clearance_margin
    if args.shield_risk_base_weight_prop_clear is not None:
        shield_override_kwargs["risk_base_weight_prop_clear"] = cfg.shield.risk_base_weight_prop_clear
    if args.shield_risk_base_weight_clear_gap is not None:
        shield_override_kwargs["risk_base_weight_clear_gap"] = cfg.shield.risk_base_weight_clear_gap
    if args.shield_risk_base_weight_support is not None:
        shield_override_kwargs["risk_base_weight_support"] = cfg.shield.risk_base_weight_support
    if args.shield_risk_base_weight_region is not None:
        shield_override_kwargs["risk_base_weight_region"] = cfg.shield.risk_base_weight_region
    if args.shield_risk_vnext2_weight_prop_clear is not None:
        shield_override_kwargs["risk_vnext2_weight_prop_clear"] = cfg.shield.risk_vnext2_weight_prop_clear
    if args.shield_risk_vnext2_weight_fragility is not None:
        shield_override_kwargs["risk_vnext2_weight_fragility"] = cfg.shield.risk_vnext2_weight_fragility
    if args.shield_risk_vnext2_weight_support is not None:
        shield_override_kwargs["risk_vnext2_weight_support"] = cfg.shield.risk_vnext2_weight_support
    if args.shield_risk_vnext2_weight_region is not None:
        shield_override_kwargs["risk_vnext2_weight_region"] = cfg.shield.risk_vnext2_weight_region
    if args.shield_risk_hist_window is not None:
        shield_override_kwargs["risk_hist_window"] = cfg.shield.risk_hist_window
    if args.shield_risk_threshold is not None:
        shield_override_kwargs["risk_threshold"] = cfg.shield.risk_threshold
    if args.shield_risk_threat_count_norm is not None:
        shield_override_kwargs["risk_threat_count_norm"] = cfg.shield.risk_threat_count_norm
    if args.shield_legacy_recursive_gate is not None:
        shield_override_kwargs["legacy_recursive_gate"] = cfg.shield.legacy_recursive_gate
    if args.shield_progressive_enabled is not None:
        shield_override_kwargs["progressive_enabled"] = cfg.shield.progressive_enabled
    if args.shield_lookahead_horizon is not None:
        shield_override_kwargs["lookahead_horizon"] = cfg.shield.lookahead_horizon
    if args.shield_risk_schedule_enabled is not None:
        shield_override_kwargs["risk_schedule_enabled"] = cfg.shield.risk_schedule_enabled

    if args.animate_checkpoint and args.animate_baseline:
        raise ValueError("Please choose either --animate-checkpoint or --animate-baseline, not both.")

    if args.ablation:
        run_ablation(args.ablation, args.device, cfg, args.algo)
        return

    eval_metrics = None
    if not args.skip_train:
        trainer = MAPPOTrainer(cfg, device=args.device) if args.algo == "mappo" else HRVDNTrainer(cfg, device=args.device)
        trainer.train(resume_path=args.resume)

    if args.validate_checkpoint:
        eval_metrics = evaluate_checkpoint(
            checkpoint_path=args.validate_checkpoint,
            episodes=args.eval_episodes,
            device=args.device,
            env_overrides=env_override_kwargs,
            shield_overrides=shield_override_kwargs,
        )
        print(f"[validate] checkpoint={args.validate_checkpoint}")
        print(f"[validate] episodes={args.eval_episodes} metrics={format_metrics(eval_metrics)}")

    if args.report:
        logdir = args.report_logdir or cfg.train.tensorboard_dir
        out = generate_training_report(
            log_dir=logdir,
            output_html=args.report_output,
            tags=parse_tags(args.report_tags),
            eval_metrics=eval_metrics,
        )
        print(f"[report] saved={out}")
        print(f"[report] open {Path(out).resolve()}")

    if args.animate_checkpoint:
        out = generate_rollout_html(
            checkpoint_path=args.animate_checkpoint,
            output_html=args.animate_output,
            device=args.device,
            max_steps=args.animate_max_steps,
            env_overrides=env_override_kwargs,
        )
        print(f"[animate] saved={out}")
        print(f"[animate] open {Path(out).resolve()}")

    if args.animate_baseline:
        out = generate_baseline_rollout_html(
            cfg=cfg,
            baseline=args.animate_baseline,
            output_html=args.animate_output,
            max_steps=args.animate_max_steps,
        )
        print(f"[animate] saved={out}")
        print(f"[animate] open {Path(out).resolve()}")


if __name__ == "__main__":
    main()
