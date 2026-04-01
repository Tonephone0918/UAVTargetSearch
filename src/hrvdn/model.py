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


class ConvMapEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128, pooled_size: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pooled_size, pooled_size)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled_size * pooled_size, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(maps))


class MAPPOActor(nn.Module):
    def __init__(self, map_channels: int, extra_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.encoder = ConvMapEncoder(map_channels, hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, maps: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        x = self.encoder(maps)
        x = torch.cat([x, extras], dim=-1)
        return self.head(x)


class MAPPOCritic(nn.Module):
    def __init__(self, state_channels: int, hidden_dim: int):
        super().__init__()
        self.encoder = ConvMapEncoder(state_channels, hidden_dim=hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        x = self.encoder(global_state)
        return self.value_head(x)
