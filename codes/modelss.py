import torch
import torch.nn as nn


def build_dqn(d_in: int, d_out: int = 5, h: int = 128, n_layers: int = 1) -> nn.Module:
    """Build a simple DQN with configurable hidden layers."""
    layers = [nn.Linear(d_in, h), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(h, h), nn.ReLU()]
    layers.append(nn.Linear(h, d_out))
    return nn.Sequential(*layers)


class DeltaNet(nn.Module):
    """Predict delta adjustment for ω from input features."""

    def __init__(self, d_in: int, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

