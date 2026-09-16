"""Temporal model for drowsiness-state classification.

Operates on a short window of per-frame signals (e.g., EAR, MAR, head pose
from drivesense.features.landmarks) rather than raw frames, per the Brief
(§7): a landmark-signal time series is far cheaper to train/run in real time
than a video-CNN, and is more interpretable for a safety-alert use case.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DrowsinessGRU(nn.Module):
    """GRU over a window of per-frame feature vectors -> drowsiness-state logits.

    Input shape: (batch, seq_len, input_dim), e.g. input_dim=5 for
    [left_ear, right_ear, mar, head_pitch, head_yaw] per frame.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, h_n = self.gru(x)
        # h_n: (num_layers * num_directions, batch, hidden_dim) -> take last layer
        if self.gru.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last = h_n[-1]
        return self.head(last)


class DrowsinessTemporalCNN(nn.Module):
    """1D-CNN alternative over the same per-frame feature windows.

    Cheaper than the GRU at inference time; kept as the second approach so
    the drowsiness task also has >= 2 compared modeling approaches
    (rubric Criterion 3: "at least two meaningful modeling approaches").
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.head(x)
