"""Small four-class CNN challenger; not a reproduction of the paper's ensemble."""

import torch
from torch import nn


class ActivityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, 31, stride=5, padding=15),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 24, 9, stride=4, padding=4),
            nn.BatchNorm1d(24),
            nn.SiLU(),
            nn.Conv1d(24, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(32, 32, 5, padding=2 * d, dilation=d),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                )
                for d in [1, 2, 4, 8]
            ]
        )
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 4))

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = x + block(x)
        return self.head(torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1))
