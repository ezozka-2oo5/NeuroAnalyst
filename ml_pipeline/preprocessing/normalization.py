"""
ml_pipeline/preprocessing/normalization.py
============================================
EEG signallarini normallashtirish (masshtablash) moduli.

Normalizatsiya o'qitish uchun numerik barqarorlikni ta'minlaydi.
Scaler parametrlari faqat train to'plamidan hisoblangan bo'lishi SHART —
test ma'lumotlari skalerni "o'rgatmasligi" kerak (data leakage oldini olish).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SignalNormalizer:
    """
    EEG epoch to'plamini turli usullarda normallashtiradi.

    Qo'llanilishi::

        norm = SignalNormalizer(method="robust")
        X_train_norm = norm.fit_transform(X_train)   # scaler fit + transform
        X_test_norm  = norm.transform(X_test)         # faqat transform

        norm.save("scaler.pkl")
        norm2 = SignalNormalizer.load("scaler.pkl")
    """

    SUPPORTED_METHODS = ("standard", "minmax", "robust", "per_epoch", "per_channel")

    def __init__(self, method: str = "robust") -> None:
        """
        Args:
            method:
                - ``"standard"``    : Z-score (mean=0, std=1). Outlierga sezgir.
                - ``"minmax"``      : [0, 1] ga keltirish. Outlierga juda sezgir.
                - ``"robust"``      : Median va IQR bilan. EEG uchun eng yaxshi.
                - ``"per_epoch"``   : Har bir epoch alohida normalized qilinadi.
                - ``"per_channel"`` : Har bir kanal alohida normalized qilinadi.
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Noto'g'ri method: '{method}'. "
                f"Tanlang: {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self._center: Optional[np.ndarray] = None  # mean yoki median
        self._scale: Optional[np.ndarray] = None   # std yoki IQR

        self._fitted = False

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "SignalNormalizer":
        """
        Train to'plamidan scaler parametrlarini hisoblaydi.

        Args:
            X: shape (N, n_channels, n_timepoints) yoki (n_channels, n_timepoints).

        Returns:
            self (method chaining uchun).
        """
        flat = self._flatten(X)  # (N*n_timepoints, n_channels)

        if self.method == "standard":
            self._center = np.mean(flat, axis=0)
            self._scale = np.std(flat, axis=0) + 1e-8

        elif self.method == "minmax":
            self._center = np.min(flat, axis=0)
            data_range = np.max(flat, axis=0) - self._center
            self._scale = np.where(data_range > 0, data_range, 1.0)

        elif self.method == "robust":
            self._center = np.median(flat, axis=0)
            q75, q25 = np.percentile(flat, [75, 25], axis=0)
            iqr = q75 - q25
            self._scale = np.where(iqr > 0, iqr, 1.0)

        elif self.method in ("per_epoch", "per_channel"):
            # Bu metodlar fit qilishni talab qilmaydi — har safar hisoblanadi
            self._center = None
            self._scale = None

        else:
            raise ValueError(f"Noma'lum method: {self.method}")

        self._fitted = True
        logger.info(f"SignalNormalizer fit qilindi: method={self.method}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Hisoblangan parametrlar bilan X ni normallashtiradi.

        Args:
            X: shape (N, n_channels, n_timepoints).

        Returns:
            Normalized X, xuddi shu shape.
        """
        if self.method == "per_epoch":
            return self._transform_per_epoch(X)
        if self.method == "per_channel":
            return self._transform_per_channel(X)

        if not self._fitted:
            raise RuntimeError("fit() ni avval chaqiring.")

        X_out = X.copy().astype(np.float32)
        n_channels = X.shape[1] if X.ndim == 3 else X.shape[0]

        # Broadcast: (1, n_channels, 1)
        center = self._center.reshape(1, n_channels, 1) if X.ndim == 3 else self._center.reshape(n_channels, 1)
        scale = self._scale.reshape(1, n_channels, 1) if X.ndim == 3 else self._scale.reshape(n_channels, 1)

        X_out = (X_out - center) / scale
        return X_out.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit() + transform() birgalikda. Faqat train to'plami uchun."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """Normalizatsiyani qaytaradi (diagnostika uchun)."""
        if self.method in ("per_epoch", "per_channel"):
            logger.warning("per_epoch/per_channel uchun inverse_transform aniq ishlamaydi.")
            return X_norm

        if not self._fitted:
            raise RuntimeError("fit() ni avval chaqiring.")

        n_channels = X_norm.shape[1] if X_norm.ndim == 3 else X_norm.shape[0]
        center = self._center.reshape(1, n_channels, 1) if X_norm.ndim == 3 else self._center.reshape(n_channels, 1)
        scale = self._scale.reshape(1, n_channels, 1) if X_norm.ndim == 3 else self._scale.reshape(n_channels, 1)

        return (X_norm * scale + center).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Scaler holatini diskka saqlaydi."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "method": self.method,
                    "center": self._center,
                    "scale": self._scale,
                    "fitted": self._fitted,
                },
                fh,
            )
        logger.info(f"Scaler saqlandi: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SignalNormalizer":
        """Diskdan scaler holatini yuklaydi."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(method=state["method"])
        obj._center = state["center"]
        obj._scale = state["scale"]
        obj._fitted = state["fitted"]
        logger.info(f"Scaler yuklandi: {path}")
        return obj

    # ──────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        """(N, C, T) → (N*T, C) shaklga keltiradi (fit uchun)."""
        if X.ndim == 2:
            return X.T  # (T, C)
        if X.ndim == 3:
            N, C, T = X.shape
            return X.transpose(0, 2, 1).reshape(-1, C)  # (N*T, C)
        raise ValueError(f"Kutilmagan shape: {X.shape}")

    @staticmethod
    def _transform_per_epoch(X: np.ndarray) -> np.ndarray:
        """Har bir epoch uchun alohida robust normalizatsiya."""
        X_out = X.copy().astype(np.float32)
        for i in range(len(X_out)):
            median = np.median(X_out[i])
            q75, q25 = np.percentile(X_out[i], [75, 25])
            iqr = q75 - q25 or 1.0
            X_out[i] = (X_out[i] - median) / iqr
        return X_out

    @staticmethod
    def _transform_per_channel(X: np.ndarray) -> np.ndarray:
        """Har bir kanal uchun alohida robust normalizatsiya."""
        X_out = X.copy().astype(np.float32)
        if X_out.ndim == 3:
            for c in range(X_out.shape[1]):
                ch_data = X_out[:, c, :]
                median = np.median(ch_data)
                q75, q25 = np.percentile(ch_data, [75, 25])
                iqr = q75 - q25 or 1.0
                X_out[:, c, :] = (ch_data - median) / iqr
        return X_out
