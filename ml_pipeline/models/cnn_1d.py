"""
ml_pipeline/models/cnn_1d.py
===============================
EEG signallari uchun 1D Convolutional Neural Network.

Raw signal (n_channels × n_timepoints) dan bevosita tasniflash.
MNE/scipy featurelari kerak emas — CNN o'zi muhim xususiyatlarni o'rganadi.

Arxitektura:
  1. Temporal convolutions (vaqt o'qi bo'yicha pattern extraction)
  2. Spatial convolutions (kanal bo'yicha ulanishlar)
  3. Depthwise separable convolutions (samaradorlik)
  4. Global Average Pooling
  5. Fully Connected + Dropout + Softmax
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import ModelConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SeparableConv1D(nn.Module):
    """
    Depthwise Separable 1D Convolution.

    Standart Conv1D ga nisbatan parametrlar soni kamligi sababli
    overfitting ga kamroq moyil va hisoblash samaraliroq.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Har kanal o'z filtrini oladi
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.elu(x)


class ResidualBlock1D(nn.Module):
    """
    Skip connection bilan Residual blok.

    Bu gradient yo'qolish muammosini (vanishing gradient) oldini oladi
    va chuqur tarmoqlarni o'qitishni osonlashtiradi.
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.25) -> None:
        super().__init__()
        padding = kernel_size // 2  # Same padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual   # Skip connection
        return F.elu(out)


class EEG1DCNN(nn.Module):
    """
    EEG tasniflash uchun 1D CNN arxitekturasi.

    Input:  (batch_size, n_channels, n_timepoints)
    Output: (batch_size, n_classes)

    Qo'llanilishi::

        model = EEG1DCNN(cfg)
        logits = model(x)                        # (B, n_classes)
        probs = torch.softmax(logits, dim=-1)    # ehtimolliklar
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        n_channels = config.n_channels
        filters = config.cnn_filters          # masalan [64, 128, 256]
        kernel = config.cnn_kernel_size       # 25 namuna ≈ ~100ms
        pool = config.cnn_pool_size           # 4
        drop = config.cnn_dropout             # 0.25
        n_classes = config.n_classes

        layers: List[nn.Module] = []

        # ── Birinchi qatlam: Temporal feature extraction ──────────
        # Input: (B, n_channels, T)
        layers.extend([
            nn.Conv1d(n_channels, filters[0], kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(filters[0]),
            nn.ELU(),
            nn.MaxPool1d(pool),
            nn.Dropout(drop),
        ])

        # ── O'rta qatlamlar: Chuqurroq featurelar ────────────────
        for i in range(1, len(filters)):
            layers.extend([
                SeparableConv1D(filters[i - 1], filters[i], kernel, padding=kernel // 2),
                ResidualBlock1D(filters[i], kernel, dropout=drop),
                nn.MaxPool1d(pool),
                nn.Dropout(drop),
            ])

        self.feature_extractor = nn.Sequential(*layers)

        # ── Global Average Pooling ────────────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)    # (B, filters[-1], 1) → (B, filters[-1])

        # ── Classifier ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1], 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(config.final_dropout),
            nn.Linear(256, n_classes),
        )

        self._initialize_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"EEG1DCNN tayyor. Parametrlar soni: {n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_timepoints) float32.

        Returns:
            logits: (batch, n_classes) float32.
        """
        features = self.feature_extractor(x)   # (B, filters[-1], T')
        pooled = self.gap(features)             # (B, filters[-1], 1)
        logits = self.classifier(pooled)        # (B, n_classes)
        return logits

    def get_feature_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classifier dan oldingi feature embeddingni qaytaradi.
        Visualization va transfer learning uchun foydali.

        Returns:
            (batch, 256) float32.
        """
        features = self.feature_extractor(x)
        pooled = self.gap(features)
        flat = self.classifier[:3](pooled)     # Linear + BN + ELU gacha
        return flat

    def _initialize_weights(self) -> None:
        """Kaiming (He) initialization — ELU aktivatsiyasi bilan mos."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
