"""Compact AlphaZero-style Residual CNN for Connect Four (Policy + Value heads)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from c4_env import COLS, NUM_ACTIONS, ROWS


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class Connect4Net(nn.Module):
    """Policy + Value ResNet taking (B, 3, 6, 7) canonical board states."""

    def __init__(self, channels: int = 64, num_blocks: int = 5) -> None:
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # Policy head: per-cell reduction -> 7 column logits
        self.policy_conv = nn.Conv2d(channels, 16, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * ROWS * COLS, NUM_ACTIONS)

        # Value head: per-cell reduction -> scalar in [-1, 1]
        self.value_conv = nn.Conv2d(channels, 8, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = torch.flatten(p, start_dim=1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = torch.flatten(v, start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return policy_logits, value


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
