from __future__ import annotations

import copy
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .config import ExperimentConfig
from .env import UAVSearchEnv
from .evaluate import evaluate
from .model import AgentQNet, VDNMixer
from .replay import SequenceReplayBuffer
from .shield import CentralizedSafetyShield
from .stats import EpisodeStatsAccumulator, SummaryCSVLogger, log_summary_scalars


def _obs_to_tensors(obs: List[Dict], device: str):
    maps = torch.stack([torch.tensor(o["map"], dtype=torch.float32) for o in obs]).to(device)
    extras = torch.stack([torch.tensor(o["extra"], dtype=torch.float32) for o in obs]).to(device)
    masks = torch.stack([torch.tensor(o["action_mask"], dtype=torch.bool) for o in obs]).to(device)
    return maps, extras, masks


class HRVDNTrainer:
    def __init__(self, cfg: ExperimentConfig, device: str = "auto"):
        self.cfg = cfg
        self.device = self._resolve_device(device)
        print(f"[HRVDNTrainer] device={self.device}")
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        random.seed(cfg.train.seed)

        self.env = UAVSearchEnv(cfg.env, cfg.reward, seed=cfg.train.seed)
        sample_obs = self.env.reset()[0]
        map_dim = int(np.prod(sample_obs["map"].shape))
        extra_dim = int(sample_obs["extra"].shape[0])
        self.policy = AgentQNet(map_dim, extra_dim, cfg.train.hidden_dim, n_actions=self.env.n_actions).to(self.device)
        self.target = copy.deepcopy(self.policy).to(self.device)
        self.mixer = VDNMixer().to(self.device)
        self.buffer = SequenceReplayBuffer(cfg.train.buffer_size)
        self.optim = optim.Adam(self.policy.parameters(), lr=cfg.train.lr_dense)
        self.ckpt_dir = Path(cfg.train.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_search_rate = float("-inf")
        self.shield = CentralizedSafetyShield(cfg)
        self.summary_logger = SummaryCSVLogger(cfg.train.tensorboard_dir)
        self.writer = None
        if cfg.train.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=cfg.train.tensorboard_dir)
                print(f"[HRVDNTrainer] tensorboard={cfg.train.tensorboard_dir}")
            except Exception as e:
                print(f"[HRVDNTrainer] tensorboard disabled ({e})")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def _pick_actions(self, obs, hs, eps):
        if not self.shield.enabled:
            actions = []
            for i, o in enumerate(obs):
                if random.random() < eps:
                    valid = np.flatnonzero(o["action_mask"])
                    actions.append(int(random.choice(valid.tolist())))
                    continue
                om = torch.tensor(o["map"], dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
                ex = torch.tensor(o["extra"], dtype=torch.float32, device=self.device).unsqueeze(0)
                q, hs[i] = self.policy(om, ex, hs[i])
                mask = torch.tensor(o["action_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
                q = q.masked_fill(~mask, -1e9)
                actions.append(int(q.argmax(dim=-1).item()))
            dummy_preferences = np.zeros((self.cfg.env.n_uavs, self.env.n_actions), dtype=np.float32)
            final_actions, _, shield_stats = self.shield.apply(
                self.env,
                actions,
                dummy_preferences,
                np.stack([o["action_mask"] for o in obs]),
                selection_mode="argmax",
            )
            return final_actions, hs, shield_stats

        actions = []
        preferences = []
        for i, o in enumerate(obs):
            om = torch.tensor(o["map"], dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
            ex = torch.tensor(o["extra"], dtype=torch.float32, device=self.device).unsqueeze(0)
            q, hs[i] = self.policy(om, ex, hs[i])
            mask = torch.tensor(o["action_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
            q = q.masked_fill(~mask, -1e9)
            preferences.append(q.squeeze(0))
            if random.random() < eps:
                valid = np.flatnonzero(o["action_mask"])
                actions.append(int(random.choice(valid.tolist())))
            else:
                actions.append(int(q.argmax(dim=-1).item()))
        final_actions, _, shield_stats = self.shield.apply(
            self.env,
            actions,
            torch.stack(preferences, dim=0),
            np.stack([o["action_mask"] for o in obs]),
            selection_mode="argmax",
        )
        return final_actions, hs, shield_stats

    def _save_checkpoint(self, name: str, epoch: int, epsilon: float, metrics: Dict[str, float] | None = None):
        path = self.ckpt_dir / name
        payload = {
            "algo": "hrvdn",
            "epoch": epoch,
            "device": self.device,
            "epsilon": epsilon,
            "reward_mode": self.cfg.reward.mode,
            "config": asdict(self.cfg),
            "policy_state_dict": self.policy.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "metrics": metrics or {},
            "best_search_rate": self.best_search_rate,
        }
        torch.save(payload, path)
        return path

    def _load_checkpoint(self, path: str, default_eps: float):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_search_rate = float(ckpt.get("best_search_rate", self.best_search_rate))
        self.env.clone_for_reward(ckpt.get("reward_mode", self.cfg.reward.mode))

        last_epoch = int(ckpt.get("epoch", -1))
        resume_epoch = max(0, last_epoch + 1)
        resume_eps = float(ckpt.get("epsilon", default_eps))
        return resume_epoch, resume_eps

    def _train_step(self):
        if len(self.buffer) < self.cfg.train.batch_size:
            return None
        batch = self.buffer.sample(self.cfg.train.batch_size)

        q_taken_list = []
        tgt_list = []

        for tr in batch:
            obs, next_obs = tr["obs"], tr["next_obs"]
            actions, reward, done = tr["actions"], tr["reward"], tr["done"]

            q_agents = []
            next_q_agents = []
            for i in range(self.cfg.env.n_uavs):
                h0 = torch.zeros(1, 1, self.cfg.train.hidden_dim, device=self.device)
                om = torch.tensor(obs[i]["map"], dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
                ex = torch.tensor(obs[i]["extra"], dtype=torch.float32, device=self.device).unsqueeze(0)
                q, _ = self.policy(om, ex, h0)
                q_agents.append(q[0, actions[i]])

                nom = torch.tensor(next_obs[i]["map"], dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
                nex = torch.tensor(next_obs[i]["extra"], dtype=torch.float32, device=self.device).unsqueeze(0)
                tq, _ = self.target(nom, nex, h0)
                nmask = torch.tensor(next_obs[i]["action_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
                tq = tq.masked_fill(~nmask, -1e9)
                next_q_agents.append(tq.max(dim=-1).values[0])

            q_total = torch.stack(q_agents).sum()
            next_q_total = torch.stack(next_q_agents).sum()
            target = torch.tensor(reward, dtype=torch.float32, device=self.device)
            if not done:
                target = target + self.cfg.train.gamma * next_q_total
            q_taken_list.append(q_total)
            tgt_list.append(target)

        q_taken = torch.stack(q_taken_list)
        q_target = torch.stack(tgt_list).detach()
        loss = F.mse_loss(q_taken, q_target)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        self.optim.step()
        return float(loss.item())

    def _make_eval_env(self) -> UAVSearchEnv:
        return UAVSearchEnv(
            copy.deepcopy(self.cfg.env),
            copy.deepcopy(self.cfg.reward),
            seed=self.cfg.train.seed + 10_000,
        )

    def train(self, resume_path: str | None = None):
        total_epochs = self.cfg.train.dense_epochs + self.cfg.train.sparse_epochs
        eps = self.cfg.train.epsilon_start
        eps_decay = (self.cfg.train.epsilon_start - self.cfg.train.epsilon_min) / max(1, self.cfg.train.dense_epochs)
        start_epoch = 0

        if resume_path:
            ckpt_path = Path(resume_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
            start_epoch, eps = self._load_checkpoint(str(ckpt_path), eps)
            print(f"[HRVDNTrainer] resume from {ckpt_path} at epoch={start_epoch} eps={eps:.4f}")

        history = []
        for ep in range(start_epoch, total_epochs):
            if ep == self.cfg.train.dense_epochs and self.cfg.reward.mode == "hybrid":
                self.env.clone_for_reward("sparse")
                self.target.load_state_dict(self.policy.state_dict())
                self.optim = optim.Adam(self.policy.parameters(), lr=self.cfg.train.lr_sparse)
                self.buffer.recalc_rewards(lambda ctx: ctx["sparse_reward"])

            obs = self.env.reset()
            self.shield.reset_episode()
            episode_stats = EpisodeStatsAccumulator()
            hs = [torch.zeros(1, 1, self.cfg.train.hidden_dim, device=self.device) for _ in range(self.cfg.env.n_uavs)]
            done = False
            ep_losses = []
            last_info = None
            episode_step_time = 0.0
            episode_stats_time = 0.0
            while not done:
                step_start = perf_counter()
                actions, hs, shield_stats = self._pick_actions(obs, hs, eps)
                next_obs, reward, done, info = self.env.step(actions)
                reward, info = self.shield.apply_reward_penalty(reward, info, shield_stats)
                stats_start = perf_counter()
                episode_stats.update(info)
                episode_stats_time += perf_counter() - stats_start

                self.buffer.push(
                    {
                        "obs": obs,
                        "actions": actions,
                        "reward": reward,
                        "next_obs": next_obs,
                        "done": done,
                        "reward_ctx": {"sparse_reward": info.get("shield_penalized_sparse_reward", info["sparse_reward"])},
                    }
                )
                obs = next_obs
                loss = self._train_step()
                if loss is not None:
                    ep_losses.append(loss)
                last_info = info
                episode_step_time += perf_counter() - step_start

            if ep < self.cfg.train.dense_epochs:
                eps = max(self.cfg.train.epsilon_min, eps - eps_decay)
            elif self.cfg.reward.mode == "sparse":
                eps = max(self.cfg.train.epsilon_min, eps * 0.999)

            if ep % self.cfg.train.target_update_interval == 0:
                self.target.load_state_dict(self.policy.state_dict())

            phase = "dense" if ep < self.cfg.train.dense_epochs else "sparse"
            episode_metrics = episode_stats.finalize(
                last_info or {},
                found_targets=len(self.env.found_targets),
                shield_mode=self.cfg.shield.mode,
            )
            episode_metrics.update(self.shield.profile_summary(episode_step_time, episode_stats_time))
            self.summary_logger.append(
                {
                    "split": "train",
                    "epoch": ep,
                    "phase": phase,
                    "mode": self.cfg.shield.mode,
                    "episodes": 1,
                    **episode_metrics,
                }
            )

            if self.writer is not None:
                log_summary_scalars(self.writer, "train", episode_metrics, ep)
                self.writer.add_scalar("train/epsilon", eps, ep)
                if ep_losses:
                    self.writer.add_scalar("train/loss", float(np.mean(ep_losses)), ep)

            mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            print(
                f"epoch={ep} phase={phase} eps={eps:.4f} return={episode_metrics['episode_return']:.3f} "
                f"orig_return={episode_metrics['original_episode_return']:.3f} loss={mean_loss:.6f} "
                f"search={episode_metrics['search_rate']:.3f} coverage={episode_metrics['coverage_ratio']:.3f} "
                f"collisions={episode_metrics['collision_count']:.0f} trigger_rate={episode_metrics['shield_trigger_rate']:.3f} "
                f"near_miss={episode_metrics['near_miss_rate']:.3f} "
                f"steps_per_sec={episode_metrics.get('perf_steps_per_sec', 0.0):.2f}"
            )

            if ep % 50 == 0:
                eval_env = self._make_eval_env()
                m = evaluate(eval_env, self.policy, episodes=2, device=self.device, shield=CentralizedSafetyShield(self.cfg))
                history.append((ep, m))
                self.summary_logger.append(
                    {
                        "split": "eval",
                        "epoch": ep,
                        "phase": phase,
                        "mode": self.cfg.shield.mode,
                        "episodes": 2,
                        **m,
                    }
                )
                print(f"epoch={ep} metrics={m}")
                if self.writer is not None:
                    log_summary_scalars(self.writer, "eval", m, ep)
                if self.cfg.train.save_best and m["search_rate"] > self.best_search_rate:
                    self.best_search_rate = m["search_rate"]
                    self._save_checkpoint("best.pt", ep, eps, m)

            if self.cfg.train.save_every > 0 and ep % self.cfg.train.save_every == 0:
                self._save_checkpoint(f"epoch_{ep}.pt", ep, eps)
                self._save_checkpoint("latest.pt", ep, eps)

        last_epoch = (total_epochs - 1) if total_epochs > 0 else 0
        self._save_checkpoint("final.pt", last_epoch, eps)
        self._save_checkpoint("latest.pt", last_epoch, eps)
        if self.writer is not None:
            self.writer.close()
        return history
