"""
ml_pipeline/models/cnn_lstm.py
================================
CNN + LSTM (BiLSTM) gibrid arxitektura — EEG tasniflash uchun.

CNN → Lokal vaqt xususiyatlarini (seizure spikes, burst-suppression) ajratadi.
LSTM → Vaqt bo'yicha uzoq muddatli bog'liqliklarni (temporal dynamics) o'rganadi.
BiLSTM → Ikki tomondan (ilgari va orqaga) o'qitish — aniqlik yuqoriroq.

Bu arxitektura CHB-MIT seizure detection da state-of-the-art natijalar beradi.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import ModelConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TemporalBlock(nn.Module):
    """
    Temporal Convolutional Block.

    Input shape: (batch, n_channels, time)
    Output shape: (batch, out_filters, time // pool)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 25,
        pool: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            nn.Conv1d(out_ch, out_ch, kernel // 2 + 1, padding=kernel // 4, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            nn.MaxPool1d(pool),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialBlock(nn.Module):
    """
    Kanal orasidagi spatial pattern larni o'rganish.

    Pointwise convolution barcha kanallarni birgalikda qaraydi —
    miyaning tarmoq ulanishlarini ifodalaydi.
    """

    def __init__(self, in_ch: int, out_ch: int, n_eeg_channels: int) -> None:
        super().__init__()
        # Depthwise — har kanal alohida
        self.depthwise = nn.Conv2d(
            1, n_eeg_channels,
            kernel_size=(n_eeg_channels, 1),
            groups=1, bias=False,
        )
        self.bn = nn.BatchNorm2d(n_eeg_channels)
        # Pointwise — kanallarni birlashtirish
        self.pointwise = nn.Conv2d(n_eeg_channels, out_ch, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_eeg_channels, T)
        Returns:
            (B, out_ch, T)
        """
        x = x.unsqueeze(1)                      # (B, 1, C, T)
        x = F.elu(self.bn(self.depthwise(x)))   # (B, C, 1, T)
        x = F.elu(self.bn2(self.pointwise(x)))  # (B, out_ch, 1, T)
        x = x.squeeze(2)                        # (B, out_ch, T)
        return x


class AttentionLayer(nn.Module):
    """
    Temporal self-attention — LSTM chiqishidagi muhim vaqt qadamlarini
    o'zgaruvchan og'irliklar bilan ajratib ko'rsatadi.

    Seizure kabi qisqa, muhim hodisalar uchun juda muhim.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size) — attention bilan og'irlangan vektor
        """
        scores = self.attention(lstm_output)           # (B, T, 1)
        weights = torch.softmax(scores, dim=1)         # (B, T, 1)
        context = torch.sum(weights * lstm_output, dim=1)  # (B, hidden)
        return context


class EEGCNNLSTM(nn.Module):
    """
    CNN + BiLSTM + Attention gibrid EEG tasniflash modeli.

    Input:  (batch_size, n_channels, n_timepoints)
    Output: (batch_size, n_classes)

    O'qitish jarayoni:
      1. CNN: Local temporal & spatial patterns
      2. BiLSTM: Long-range temporal dependencies
      3. Attention: Muhim vaqt qadamlarini ajratish
      4. FC: Tasniflash

    Qo'llanilishi::

        model = EEGCNNLSTM(cfg)
        logits = model(x)                        # (B, n_classes)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        n_ch = config.n_channels
        filters = config.cnn_filters          # [64, 128, 256]
        kernel = config.cnn_kernel_size       # 25
        pool = config.cnn_pool_size           # 4
        cnn_drop = config.cnn_dropout         # 0.25

        lstm_units = config.lstm_units        # 128
        lstm_layers = config.lstm_layers      # 2
        lstm_drop = config.lstm_dropout       # 0.3
        bidirectional = config.lstm_bidirectional  # True
        n_classes = config.n_classes

        # ── CNN bloklari ──────────────────────────────────────────
        # Birinchi blok: Temporal extraction
        self.temporal_block1 = TemporalBlock(n_ch, filters[0], kernel, pool, cnn_drop)
        self.temporal_block2 = TemporalBlock(filters[0], filters[1], kernel // 2 + 1, pool, cnn_drop)

        if len(filters) > 2:
            self.temporal_block3 = TemporalBlock(filters[1], filters[2], kernel // 4 + 1, 2, cnn_drop)
            cnn_out_ch = filters[2]
        else:
            self.temporal_block3 = None
            cnn_out_ch = filters[1]

        # ── BiLSTM qatlami ────────────────────────────────────────
        # CNN chiqishi: (B, cnn_out_ch, T') → LSTM input: (B, T', cnn_out_ch)
        lstm_hidden = lstm_units
        lstm_out_size = lstm_hidden * 2 if bidirectional else lstm_hidden

        self.lstm = nn.LSTM(
            input_size=cnn_out_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_drop if lstm_layers > 1 else 0.0,
        )

        # ── Attention ────────────────────────────────────────────
        self.attention = AttentionLayer(lstm_out_size)

        # ── Layer Normalization ───────────────────────────────────
        self.layer_norm = nn.LayerNorm(lstm_out_size)

        # ── Classifier ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(config.final_dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(config.final_dropout / 2),
            nn.Linear(128, n_classes),
        )

        self._initialize_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"EEGCNNLSTM tayyor: "
            f"CNN({','.join(map(str, filters))}) + "
            f"{'Bi' if bidirectional else ''}LSTM({lstm_hidden}x{lstm_layers}) + "
            f"Attention → {n_classes} class. "
            f"Parametrlar: {n_params:,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_channels, n_timepoints) float32.
            return_attention: True bo'lsa attention weights ham qaytariladi.

        Returns:
            logits: (batch, n_classes)
            yoki
            (logits, attention_weights) agar return_attention=True.
        """
        # ── CNN ───────────────────────────────────────────────────
        out = self.temporal_block1(x)    # (B, filters[0], T1)
        out = self.temporal_block2(out)  # (B, filters[1], T2)
        if self.temporal_block3 is not None:
            out = self.temporal_block3(out)  # (B, filters[2], T3)

        # CNN chiqishini LSTM uchun shape'ga o'tkazish
        out = out.permute(0, 2, 1)       # (B, T3, cnn_out_ch)

        # ── LSTM ──────────────────────────────────────────────────
        lstm_out, _ = self.lstm(out)     # (B, T3, lstm_out_size)
        lstm_out = self.layer_norm(lstm_out)

        # ── Attention ─────────────────────────────────────────────
        context = self.attention(lstm_out)  # (B, lstm_out_size)

        # ── Classifier ────────────────────────────────────────────
        logits = self.classifier(context)   # (B, n_classes)

        return logits

    def _initialize_weights(self) -> None:
        """Og'irliklarni boshlang'ich qiymatlar bilan to'ldirish."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)
                        # Forget gate biasini 1 ga o'rnatish (LSTM trick)
                        n = param.size(0)
                        param.data[n // 4: n // 2].fill_(1.0)
