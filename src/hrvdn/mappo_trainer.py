from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical

from .config import ExperimentConfig
from .env import UAVSearchEnv
from .evaluate import evaluate_actor_policy
from .runtime import build_mappo_from_env


class MAPPOTrainer:
    def __init__(self, cfg: ExperimentConfig, device: str = "auto"):
        self.cfg = cfg
        self.device = self._resolve_device(device)
        print(f"[MAPPOTrainer] device={self.device}")
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        random.seed(cfg.train.seed)

        self.env = UAVSearchEnv(cfg.env, cfg.reward, seed=cfg.train.seed)
        self.actor, self.critic = build_mappo_from_env(cfg, self.env, self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.mappo.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.mappo.critic_lr)
        self.ckpt_dir = Path(cfg.train.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_search_rate = float("-inf")
        self.writer = None
        if cfg.train.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=cfg.train.tensorboard_dir)
                print(f"[MAPPOTrainer] tensorboard={cfg.train.tensorboard_dir}")
            except Exception as e:
                print(f"[MAPPOTrainer] tensorboard disabled ({e})")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def _set_phase_lr(self, sparse: bool) -> None:
        actor_lr = self.cfg.mappo.actor_lr_sparse if sparse else self.cfg.mappo.actor_lr
        critic_lr = self.cfg.mappo.critic_lr_sparse if sparse else self.cfg.mappo.critic_lr
        for group in self.actor_optim.param_groups:
            group["lr"] = actor_lr
        for group in self.critic_optim.param_groups:
            group["lr"] = critic_lr

    def _in_sparse_phase(self, epoch: int) -> bool:
        if self.cfg.reward.mode == "sparse":
            return True
        if self.cfg.reward.mode == "dense":
            return False
        return epoch >= self.cfg.train.dense_epochs

    def _make_eval_env(self) -> UAVSearchEnv:
        return UAVSearchEnv(
            deepcopy(self.cfg.env),
            deepcopy(self.cfg.reward),
            seed=self.cfg.train.seed + 10_000,
        )

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict[str, float] | None = None) -> Path:
        path = self.ckpt_dir / name
        payload = {
            "algo": "mappo",
            "epoch": epoch,
            "device": self.device,
            "reward_mode": self.cfg.reward.mode,
            "config": asdict(self.cfg),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optim.state_dict(),
            "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            "metrics": metrics or {},
            "best_search_rate": self.best_search_rate,
        }
        torch.save(payload, path)
        return path

    def _load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_optim.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_optim.load_state_dict(ckpt["critic_optimizer_state_dict"])
        self.best_search_rate = float(ckpt.get("best_search_rate", self.best_search_rate))
        self.env.clone_for_reward(ckpt.get("reward_mode", self.cfg.reward.mode))
        return max(0, int(ckpt.get("epoch", -1)) + 1)

    def _collect_episode(self) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
        obs = self.env.reset()
        done = False
        ep_reward = 0.0
        last_info: Dict[str, float] = {}

        obs_maps = []
        obs_extras = []
        action_masks = []
        global_states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        while not done:
            maps = torch.tensor(np.stack([o["map"] for o in obs]), dtype=torch.float32, device=self.device)
            extras = torch.tensor(np.stack([o["extra"] for o in obs]), dtype=torch.float32, device=self.device)
            masks = torch.tensor(np.stack([o["action_mask"] for o in obs]), dtype=torch.bool, device=self.device)
            state = torch.tensor(self.env.global_state(), dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.actor(maps, extras)
            logits = logits.masked_fill(~masks, -1e9)
            dist = Categorical(logits=logits)
            act = dist.sample()
            log_prob = dist.log_prob(act)
            value = self.critic(state).squeeze(0)

            next_obs, reward, done, info = self.env.step(act.tolist())

            obs_maps.append(maps.cpu().numpy())
            obs_extras.append(extras.cpu().numpy())
            action_masks.append(masks.cpu().numpy())
            global_states.append(state.squeeze(0).cpu().numpy())
            actions.append(act.detach().cpu().numpy())
            log_probs.append(log_prob.detach().cpu().numpy())
            values.append(float(value.item()))
            rewards.append(float(reward))
            dones.append(float(done))

            ep_reward += reward
            obs = next_obs
            last_info = info

        batch = {
            "obs_maps": np.asarray(obs_maps, dtype=np.float32),
            "obs_extras": np.asarray(obs_extras, dtype=np.float32),
            "action_masks": np.asarray(action_masks, dtype=np.bool_),
            "global_states": np.asarray(global_states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "log_probs": np.asarray(log_probs, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }
        info_out = dict(last_info)
        info_out["episode_reward"] = float(ep_reward)
        info_out["episode_steps"] = float(len(rewards))
        info_out["collisions"] = float(self.env.collisions)
        return batch, info_out

    def _compute_returns_and_advantages(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        next_value = 0.0
        gamma = self.cfg.train.gamma
        lam = self.cfg.mappo.gae_lambda
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            advantages[t] = last_gae
            next_value = values[t]
        returns = advantages + values
        return returns.astype(np.float32), advantages.astype(np.float32)

    def _ppo_update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        returns, advantages = self._compute_returns_and_advantages(batch["rewards"], batch["values"], batch["dones"])
        adv_norm = (advantages - advantages.mean()) / max(1e-8, advantages.std())

        t_steps, n_agents = batch["actions"].shape[:2]
        obs_maps = torch.tensor(batch["obs_maps"].reshape(t_steps * n_agents, *batch["obs_maps"].shape[2:]), dtype=torch.float32, device=self.device)
        obs_extras = torch.tensor(
            batch["obs_extras"].reshape(t_steps * n_agents, batch["obs_extras"].shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        action_masks = torch.tensor(
            batch["action_masks"].reshape(t_steps * n_agents, batch["action_masks"].shape[-1]),
            dtype=torch.bool,
            device=self.device,
        )
        actions = torch.tensor(batch["actions"].reshape(-1), dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(batch["log_probs"].reshape(-1), dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(np.repeat(adv_norm, n_agents), dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_values_t = torch.tensor(batch["values"], dtype=torch.float32, device=self.device)
        global_states = torch.tensor(batch["global_states"], dtype=torch.float32, device=self.device)

        actor_losses = []
        critic_losses = []
        entropies = []
        approx_kls = []

        for _ in range(self.cfg.mappo.update_epochs):
            logits = self.actor(obs_maps, obs_extras)
            logits = logits.masked_fill(~action_masks, -1e9)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.mappo.clip_coef, 1.0 + self.cfg.mappo.clip_coef) * advantages_t
            actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.mappo.entropy_coef * entropy

            values_pred = self.critic(global_states).squeeze(-1)
            value_clipped = old_values_t + torch.clamp(
                values_pred - old_values_t,
                -self.cfg.mappo.clip_coef,
                self.cfg.mappo.clip_coef,
            )
            value_loss_unclipped = (values_pred - returns_t) ** 2
            value_loss_clipped = (value_clipped - returns_t) ** 2
            critic_loss = self.cfg.mappo.value_coef * 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.mappo.max_grad_norm)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.mappo.max_grad_norm)
            self.critic_optim.step()

            actor_losses.append(float(actor_loss.item()))
            critic_losses.append(float(critic_loss.item()))
            entropies.append(float(entropy.item()))
            approx_kl = float((old_log_probs - new_log_probs).mean().item())
            approx_kls.append(approx_kl)
            if self.cfg.mappo.target_kl > 0 and approx_kl > self.cfg.mappo.target_kl:
                break

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }

    def train(self, resume_path: str | None = None):
        total_epochs = self.cfg.train.dense_epochs + self.cfg.train.sparse_epochs
        start_epoch = 0

        if resume_path:
            ckpt_path = Path(resume_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
            start_epoch = self._load_checkpoint(str(ckpt_path))
            print(f"[MAPPOTrainer] resume from {ckpt_path} at epoch={start_epoch}")

        history = []
        for ep in range(start_epoch, total_epochs):
            if ep == self.cfg.train.dense_epochs and self.cfg.reward.mode == "hybrid":
                self.env.clone_for_reward("sparse")
                self._set_phase_lr(sparse=True)
            elif ep == start_epoch:
                self._set_phase_lr(sparse=self._in_sparse_phase(ep))

            batch, info = self._collect_episode()
            losses = self._ppo_update(batch)

            if self.writer is not None:
                self.writer.add_scalar("train/episode_reward", info["episode_reward"], ep)
                self.writer.add_scalar("train/search_rate", info["search_rate"], ep)
                self.writer.add_scalar("train/coverage_rate", info["coverage_rate"], ep)
                self.writer.add_scalar("train/collisions", info["collisions"], ep)
                self.writer.add_scalar("train/episode_steps", info["episode_steps"], ep)
                self.writer.add_scalar("train/actor_loss", losses["actor_loss"], ep)
                self.writer.add_scalar("train/critic_loss", losses["critic_loss"], ep)
                self.writer.add_scalar("train/entropy", losses["entropy"], ep)
                self.writer.add_scalar("train/approx_kl", losses["approx_kl"], ep)

            phase = "dense" if ep < self.cfg.train.dense_epochs else "sparse"
            print(
                f"epoch={ep} phase={phase} reward={info['episode_reward']:.3f} "
                f"actor_loss={losses['actor_loss']:.6f} critic_loss={losses['critic_loss']:.6f} "
                f"search={info['search_rate']:.3f} coverage={info['coverage_rate']:.3f} "
                f"collisions={info['collisions']}"
            )

            if ep % 50 == 0:
                eval_env = self._make_eval_env()
                m = evaluate_actor_policy(eval_env, self.actor, episodes=5, device=self.device)
                history.append((ep, m))
                print(f"epoch={ep} metrics={m}")
                if self.writer is not None:
                    self.writer.add_scalar("eval/search_rate", m["search_rate"], ep)
                    self.writer.add_scalar("eval/coverage_rate", m["coverage_rate"], ep)
                    self.writer.add_scalar("eval/collisions", m["collisions"], ep)
                    self.writer.add_scalar("eval/avg_reward", m["avg_reward"], ep)
                    self.writer.add_scalar("eval/error_rate", m["error_rate"], ep)
                if self.cfg.train.save_best and m["search_rate"] > self.best_search_rate:
                    self.best_search_rate = m["search_rate"]
                    self._save_checkpoint("best.pt", ep, m)

            if self.cfg.train.save_every > 0 and ep % self.cfg.train.save_every == 0:
                self._save_checkpoint(f"epoch_{ep}.pt", ep)
                self._save_checkpoint("latest.pt", ep)

        last_epoch = (total_epochs - 1) if total_epochs > 0 else 0
        self._save_checkpoint("final.pt", last_epoch)
        self._save_checkpoint("latest.pt", last_epoch)
        if self.writer is not None:
            self.writer.close()
        return history
