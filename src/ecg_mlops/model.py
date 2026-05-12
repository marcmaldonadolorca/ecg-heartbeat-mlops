from __future__ import annotations

import torch
import torch.nn as nn


class ECGCNN(nn.Module):
    def __init__(self, n_classes: int = 5, base_filters: int = 32, dropout: float = 0.10):
        super().__init__()
        c1 = base_filters
        c2 = base_filters * 2
        c3 = base_filters * 4

        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout / 2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(c3, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, 64),
            nn.ReLU(),
            nn.Dropout(dropout + 0.05),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

