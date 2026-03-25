from __future__ import annotations

import copy
import random
from dataclasses import asdict
from pathlib import Path
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
        self.policy = AgentQNet(map_dim, extra_dim, cfg.train.hidden_dim).to(self.device)
        self.target = copy.deepcopy(self.policy).to(self.device)
        self.mixer = VDNMixer().to(self.device)
        self.buffer = SequenceReplayBuffer(cfg.train.buffer_size)
        self.optim = optim.Adam(self.policy.parameters(), lr=cfg.train.lr_dense)
        self.ckpt_dir = Path(cfg.train.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_search_rate = float("-inf")
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
        return actions, hs

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict[str, float] | None = None):
        path = self.ckpt_dir / name
        payload = {
            "epoch": epoch,
            "device": self.device,
            "config": asdict(self.cfg),
            "policy_state_dict": self.policy.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "metrics": metrics or {},
            "best_search_rate": self.best_search_rate,
        }
        torch.save(payload, path)
        return path

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

    def train(self):
        total_epochs = self.cfg.train.dense_epochs + self.cfg.train.sparse_epochs
        eps = self.cfg.train.epsilon_start
        eps_decay = (self.cfg.train.epsilon_start - self.cfg.train.epsilon_min) / max(1, self.cfg.train.dense_epochs)

        history = []
        for ep in range(total_epochs):
            if ep == self.cfg.train.dense_epochs and self.cfg.reward.mode == "hybrid":
                self.env.clone_for_reward("sparse")
                self.target.load_state_dict(self.policy.state_dict())
                self.optim = optim.Adam(self.policy.parameters(), lr=self.cfg.train.lr_sparse)
                self.buffer.recalc_rewards(lambda ctx: ctx["sparse_reward"])

            obs = self.env.reset()
            hs = [torch.zeros(1, 1, self.cfg.train.hidden_dim, device=self.device) for _ in range(self.cfg.env.n_uavs)]
            done = False
            ep_reward = 0.0
            ep_losses = []
            while not done:
                actions, hs = self._pick_actions(obs, hs, eps)
                next_obs, reward, done, info = self.env.step(actions)
                ep_reward += reward

                self.buffer.push(
                    {
                        "obs": obs,
                        "actions": actions,
                        "reward": reward,
                        "next_obs": next_obs,
                        "done": done,
                        "reward_ctx": {"sparse_reward": info["sparse_reward"]},
                    }
                )
                obs = next_obs
                loss = self._train_step()
                if loss is not None:
                    ep_losses.append(loss)

            if ep < self.cfg.train.dense_epochs:
                eps = max(self.cfg.train.epsilon_min, eps - eps_decay)
            elif self.cfg.reward.mode == "sparse":
                eps = max(self.cfg.train.epsilon_min, eps * 0.999)

            if ep % self.cfg.train.target_update_interval == 0:
                self.target.load_state_dict(self.policy.state_dict())

            if self.writer is not None:
                self.writer.add_scalar("train/episode_reward", ep_reward, ep)
                self.writer.add_scalar("train/epsilon", eps, ep)
                self.writer.add_scalar("train/search_rate", info["search_rate"], ep)
                self.writer.add_scalar("train/coverage_rate", info["coverage_rate"], ep)
                self.writer.add_scalar("train/collisions", info["collisions"], ep)
                if ep_losses:
                    self.writer.add_scalar("train/loss", float(np.mean(ep_losses)), ep)

            if ep % 50 == 0:
                m = evaluate(self.env, self.policy, episodes=2, device=self.device)
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

        self._save_checkpoint("final.pt", total_epochs - 1 if total_epochs > 0 else 0)
        self._save_checkpoint("latest.pt", total_epochs - 1 if total_epochs > 0 else 0)
        if self.writer is not None:
            self.writer.close()
        return history
