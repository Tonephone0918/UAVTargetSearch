from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class EnvConfig:
    map_size: int = 20
    n_uavs: int = 10
    n_targets: int = 10
    n_threats: int = 5
    comm_radius: int = 5
    uav_safe_dist: int = 1
    threat_safe_dist: int = 2

    n_altitudes: int = 3
    pd_levels: List[float] = field(default_factory=lambda: [0.94, 0.82, 0.70])
    pf_levels: List[float] = field(default_factory=lambda: [0.05, 0.18, 0.30])
    sense_radii: List[int] = field(default_factory=lambda: [1, 2, 3])
    max_turn_rad: float = 0.78539816339  # pi/4

    target_speed: float = 0.2
    target_alpha: float = 0.5
    dynamic_threat: bool = True
    threat_move_period: int = 3

    t0: int = 20
    zeta_p: float = 0.75
    xi_p: float = 0.99
    ea: float = 0.45
    ga: float = 0.3
    da: float = 2.0
    p_delta: float = 0.5

    patch_size: int = 11
    max_steps: int = 120
    strict_found_detection: bool = True
    terminate_on_all_targets_found: bool = False


@dataclass
class RewardConfig:
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    r1_discovery: float = 15.0
    r2_time: float = -1.0
    r3_collision: float = -5.0
    r4_entropy: float = 0.5
    r5_pheromone: float = 0.1
    r6_coverage: float = 1.0
    r7_energy_penalty: float = 5.0

    use_compensation: bool = True
    use_energy_penalty: bool = True
    normalize_dpm_reward: bool = False


@dataclass
class TrainConfig:
    buffer_size: int = 900
    batch_size: int = 20
    hidden_dim: int = 96
    grad_clip: float = 10.0

    dense_epochs: int = 600
    sparse_epochs: int = 2800
    lr_dense: float = 0.005
    lr_sparse: float = 0.0005
    gamma: float = 0.99

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    target_update_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    save_every: int = 50
    save_best: bool = True
    tensorboard_dir: str = "runs/hrvdn"
    use_tensorboard: bool = True

    seed: int = 42


@dataclass
class MappoConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    actor_lr_sparse: float = 1e-4
    critic_lr_sparse: float = 3e-4
    update_epochs: int = 4
    num_minibatches: int = 4
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03


@dataclass
class ShieldConfig:
    enabled: bool = False
    mode: Literal["off", "safe", "recursive"] = "off"
    log_stats: bool = True
    penalty_coef: float = 1.0
    near_miss_margin: float = 0.5
    profile_enabled: bool = False
    cache_enabled: bool = True
    refine_enabled: bool = True
    refine_margin: float = 0.5
    candidate_top_k: int = 2
    candidate_full_fallback: bool = False
    local_uav_padding: float = 1.0
    local_threat_padding: float = 1.0
    threat_radius_inflation: float = 0.0
    recursive_margin_threshold: float = 0.25
    recursive_safe_action_threshold: int = 2
    recursive_recent_window: int = 5
    recursive_recent_trigger_threshold: int = 5
    risk_score_enabled: bool = True
    risk_weight_clear: float = 0.5
    risk_weight_region: float = 0.3
    risk_weight_hist: float = 0.2
    risk_clearance_norm: float = 1.0
    risk_hist_window: int = 5
    risk_threshold: float = 0.5
    risk_threat_count_norm: float = 3.0
    legacy_recursive_gate: bool = False

    # Reserved interface for later stages.
    progressive_enabled: bool = False
    lookahead_horizon: int = 1
    risk_schedule_enabled: bool = False


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mappo: MappoConfig = field(default_factory=MappoConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
