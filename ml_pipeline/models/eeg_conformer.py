"""
ml_pipeline/models/eeg_conformer.py
======================================
EEG-Conformer: CNN + Multi-Head Self-Attention Transformer

Zamonaviy Foundation Model uslubidagi arxitektura.
"EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization"
(Song et al., 2023) asosida.

CNN → Lokal vaqt xususiyatlari (short-range)
Transformer → Global kontekst (long-range dependencies)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import ModelConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PatchEmbedding(nn.Module):
    """
    EEG signalini "patch"larga bo'lib, Transformer uchun
    token embeddingga aylantiradi.

    CNN bilan EEG dan lokal featurelar ajratiladi,
    keyin ular Transformer uchun sequence bo'lib uzatiladi.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        kernel_size: int = 25,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Temporal convolution — lokal xususiyatlar
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, kernel_size // 2), bias=False),
            nn.BatchNorm2d(40),
            nn.ELU(),
        )
        # Spatial convolution — kanallar orasidagi ulanish
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=(n_channels, 1), groups=40, bias=False),
            nn.Conv2d(40, d_model, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_channels, T)

        Returns:
            (B, seq_len, d_model) — Transformer input
        """
        x = x.unsqueeze(1)                    # (B, 1, C, T)
        x = self.temporal_conv(x)             # (B, 40, C, T)
        x = self.spatial_conv(x)              # (B, d_model, 1, T//4)
        x = x.squeeze(2)                      # (B, d_model, T//4)
        x = x.permute(0, 2, 1)               # (B, T//4, d_model) — sequence
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.

    Transformer o'zi ketma-ketlik tartibini bilmaydi.
    Bu qatlam har bir token ga o'z pozitsiyasini bildiradi.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # (max_len, d_model) shaklida sabit pozitsion encoding
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            (B, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Standart Transformer Encoder bloki.

    Multi-Head Self-Attention + Feed-Forward Network.
    Pre-LayerNorm (Post-LN ga nisbatan barqarorroq o'qitiladi).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention (Pre-LN)
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = residual + attn_out

        # Feed-Forward (Pre-LN)
        residual = x
        x = residual + self.ff(self.norm2(x))
        return x


class EEGConformer(nn.Module):
    """
    EEG-Conformer: CNN + Transformer gibrid model.

    Input:  (batch_size, n_channels, n_timepoints)
    Output: (batch_size, n_classes)

    Kuchli tomonlari:
    - CNN: EEG spiklar va lokal pattern lar uchun
    - Transformer: Uzoq muddatli temporal context
    - Miqyoslanuvchi: Katta datasetlar bilan yaxshi o'rganadi

    Qo'llanilishi::

        model = EEGConformer(cfg)
        logits = model(x)                        # (B, n_classes)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        d_model = config.conformer_d_model     # 256
        n_heads = config.conformer_n_heads     # 8
        n_layers = config.conformer_n_layers   # 6
        ff_dim = config.conformer_ff_dim       # 1024
        drop = config.conformer_dropout        # 0.1
        n_classes = config.n_classes
        n_channels = config.n_channels

        # ── Patch Embedding (CNN) ─────────────────────────────────
        self.patch_embedding = PatchEmbedding(
            n_channels=n_channels,
            d_model=d_model,
            kernel_size=config.cnn_kernel_size,
            dropout=drop,
        )

        # ── Positional Encoding ───────────────────────────────────
        self.pos_encoding = PositionalEncoding(d_model, dropout=drop)

        # ── Transformer Encoder ───────────────────────────────────
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ff_dim, drop)
            for _ in range(n_layers)
        ])

        # ── [CLS] token ───────────────────────────────────────────
        # GPT/BERT uslubida: bitta "classification token"
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Final normalization va classifier ────────────────────
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(config.final_dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"EEGConformer tayyor: "
            f"CNN+{n_layers}xTransformer(d={d_model}, heads={n_heads}) → {n_classes} class. "
            f"Parametrlar: {n_params:,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_timepoints) float32.

        Returns:
            logits: (batch, n_classes) float32.
        """
        B = x.size(0)

        # ── Patch embedding (CNN) ────────────────────────────────
        tokens = self.patch_embedding(x)       # (B, seq_len, d_model)

        # ── CLS token qo'shish ───────────────────────────────────
        cls = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+seq_len, d_model)

        # ── Positional encoding ──────────────────────────────────
        tokens = self.pos_encoding(tokens)

        # ── Transformer blocks ───────────────────────────────────
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # ── CLS token chiqishi ────────────────────────────────────
        cls_out = self.norm(tokens[:, 0])      # (B, d_model)
        logits = self.classifier(cls_out)       # (B, n_classes)
        return logits

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """ViT uslubida trunc_normal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
