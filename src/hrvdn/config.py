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

    t0: int = 5
    zeta_p: float = 0.75
    xi_p: float = 0.99
    ea: float = 0.45
    ga: float = 0.3
    da: float = 2.0
    p_delta: float = 0.5

    patch_size: int = 11
    max_steps: int = 80


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
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
