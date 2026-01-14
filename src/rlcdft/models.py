from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes, act_last=None):
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if act_last == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Scalar-action actor (tanh output in [-1, 1])."""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.body = mlp([state_dim, hidden, hidden, 1], act_last="tanh")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.body(s).squeeze(1)


class Critic(nn.Module):
    """Twin Q critics for TD3, taking (state, action)."""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.q1 = mlp([state_dim + 1, hidden, hidden, 1])
        self.q2 = mlp([state_dim + 1, hidden, hidden, 1])

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, a], dim=1)
        return self.q1(x).squeeze(1), self.q2(x).squeeze(1)
