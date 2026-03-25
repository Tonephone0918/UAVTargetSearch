from __future__ import annotations

import torch
import torch.nn as nn


class AgentQNet(nn.Module):
    def __init__(self, input_dim: int, extra_dim: int, hidden_dim: int = 96, n_actions: int = 12):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs_flat: torch.Tensor, extra: torch.Tensor, h: torch.Tensor):
        x = torch.cat([obs_flat, extra], dim=-1)
        x = self.enc(x)
        x = x.unsqueeze(1)
        y, h_next = self.gru(x, h)
        q = self.head(y[:, -1, :])
        return q, h_next


class VDNMixer(nn.Module):
    def forward(self, q_agents: torch.Tensor):
        # q_agents: [B, N]
        return q_agents.sum(dim=-1, keepdim=True)
