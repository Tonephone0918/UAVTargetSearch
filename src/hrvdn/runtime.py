from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .config import EnvConfig, ExperimentConfig, MappoConfig, RewardConfig, ShieldConfig, TrainConfig
from .env import UAVSearchEnv
from .model import AgentQNet, MAPPOActor, MAPPOCritic


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def config_from_dict(raw_cfg: Dict[str, Any]) -> ExperimentConfig:
    env_cfg = EnvConfig(**raw_cfg.get("env", {}))
    rew_cfg = RewardConfig(**raw_cfg.get("reward", {}))
    train_cfg = TrainConfig(**raw_cfg.get("train", {}))
    mappo_cfg = MappoConfig(**raw_cfg.get("mappo", {}))
    shield_cfg = ShieldConfig(**raw_cfg.get("shield", {}))
    return ExperimentConfig(env=env_cfg, reward=rew_cfg, train=train_cfg, mappo=mappo_cfg, shield=shield_cfg)


def apply_env_overrides(
    cfg: ExperimentConfig,
    *,
    map_size: int | None = None,
    n_uavs: int | None = None,
    n_targets: int | None = None,
    n_threats: int | None = None,
    max_steps: int | None = None,
    terminate_on_all_targets_found: bool | None = None,
    seed: int | None = None,
) -> ExperimentConfig:
    if map_size is not None:
        cfg.env.map_size = map_size
    if n_uavs is not None:
        cfg.env.n_uavs = n_uavs
    if n_targets is not None:
        cfg.env.n_targets = n_targets
    if n_threats is not None:
        cfg.env.n_threats = n_threats
    if max_steps is not None:
        cfg.env.max_steps = max_steps
    if terminate_on_all_targets_found is not None:
        cfg.env.terminate_on_all_targets_found = terminate_on_all_targets_found
    if seed is not None:
        cfg.train.seed = seed
    return cfg


def build_policy_from_env(cfg: ExperimentConfig, env: UAVSearchEnv, device: str) -> AgentQNet:
    sample_obs = env.reset()[0]
    map_dim = int(np.prod(sample_obs["map"].shape))
    extra_dim = int(sample_obs["extra"].shape[0])
    return AgentQNet(map_dim, extra_dim, cfg.train.hidden_dim, n_actions=env.n_actions).to(device)


def build_mappo_from_env(
    cfg: ExperimentConfig,
    env: UAVSearchEnv,
    device: str,
) -> tuple[MAPPOActor, MAPPOCritic]:
    sample_obs = env.reset()[0]
    map_channels = int(sample_obs["map"].shape[0])
    extra_dim = int(sample_obs["extra"].shape[0])
    state_channels = int(env.global_state().shape[0])
    actor = MAPPOActor(map_channels, extra_dim, cfg.train.hidden_dim, n_actions=env.n_actions).to(device)
    critic = MAPPOCritic(state_channels, cfg.train.hidden_dim).to(device)
    return actor, critic


def load_checkpoint_policy(policy: AgentQNet, state_dict: Dict[str, Any], checkpoint_path: str) -> None:
    try:
        policy.load_state_dict(state_dict)
    except RuntimeError as e:
        ckpt_name = Path(checkpoint_path).name
        raise RuntimeError(
            f"Checkpoint {ckpt_name} is incompatible with the current observation/network definition. "
            "Please retrain the model with the updated reproduction code before evaluating or replaying it."
        ) from e


def load_checkpoint_module(module: torch.nn.Module, state_dict: Dict[str, Any], checkpoint_path: str, module_name: str) -> None:
    try:
        module.load_state_dict(state_dict)
    except RuntimeError as e:
        ckpt_name = Path(checkpoint_path).name
        raise RuntimeError(
            f"Checkpoint {ckpt_name} is incompatible with the current {module_name} definition. "
            "Please retrain the model with the updated reproduction code before evaluating or replaying it."
        ) from e
