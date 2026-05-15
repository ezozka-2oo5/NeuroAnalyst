"""
ml_pipeline/features/connectivity.py
========================================
EEG kanallar orasidagi ulanishni (connectivity) o'lchovchi featurelar.

Coherence, Phase Locking Value (PLV) va Correlation matrix asosida
miyaning turli hududlari qanchalik sinxron ishlashini ifodalaydi.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import coherence as scipy_coherence

from ..utils.config import FeatureConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ConnectivityFeatureExtractor:
    """
    Kanallar orasidagi ulanish featurelarini hisoblovchi klass.

    Featurelar:
    - Pearson correlation matrix (upper triangle)
    - Spectral coherence (band-specific)
    - Phase Locking Value (PLV)

    Qo'llanilishi::

        extractor = ConnectivityFeatureExtractor(cfg)
        features = extractor.extract(epoch, sfreq=256.0)
    """

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def extract(self, epoch: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Epoch dan connectivity featurelarini ajratib oladi.

        Args:
            epoch: (n_channels, n_timepoints) float32.
            sfreq: Hz.

        Returns:
            1D feature vector, float32.
        """
        all_features: List[np.ndarray] = []

        # 1. Pearson correlation (upper triangle)
        corr_features = self._correlation_features(epoch)
        all_features.append(corr_features)

        # 2. Band-specific coherence
        coh_features = self._coherence_features(epoch, sfreq)
        all_features.append(coh_features)

        # 3. PLV (Phase Locking Value)
        plv_features = self._plv_features(epoch, sfreq)
        all_features.append(plv_features)

        return np.concatenate(all_features).astype(np.float32)

    def extract_batch(
        self, epochs: np.ndarray, sfreq: float
    ) -> np.ndarray:
        """Batch extraction."""
        results = [self.extract(epochs[i], sfreq) for i in range(len(epochs))]
        return np.stack(results, axis=0).astype(np.float32)

    # ──────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _correlation_features(epoch: np.ndarray) -> np.ndarray:
        """
        Kanal juftlari orasidagi Pearson korrelyatsiyasini hisoblaydi.

        n_channels kanallar uchun: n*(n-1)/2 ta noyob qiymat.
        """
        corr_matrix = np.corrcoef(epoch)          # (C, C)
        # Yuqori uchburchak (diagonal dan tashqari)
        upper_idx = np.triu_indices(corr_matrix.shape[0], k=1)
        upper_vals = corr_matrix[upper_idx]
        # NaN qiymatlarni 0 ga almashtirish
        upper_vals = np.nan_to_num(upper_vals, nan=0.0)
        return upper_vals.astype(np.float32)

    def _coherence_features(
        self, epoch: np.ndarray, sfreq: float
    ) -> np.ndarray:
        """
        Har bir band uchun kanal juftlari orasidagi kogerentlikni hisoblaydi.

        Band-averaged coherence — ikkita kanal bir chastota oralig'ida
        qanchalik sinxron ekanligini ko'rsatadi.
        """
        n_channels = epoch.shape[0]
        bands = self.config.bands
        features: List[float] = []

        nperseg = min(256, epoch.shape[1] // 2)

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                try:
                    freqs, coh = scipy_coherence(epoch[i], epoch[j], fs=sfreq, nperseg=nperseg)
                except Exception:
                    # Uzunlik yetarli bo'lmasa
                    features.extend([0.0] * len(bands))
                    continue

                for band_name, (low, high) in bands.items():
                    mask = (freqs >= low) & (freqs < high)
                    if np.any(mask):
                        features.append(float(np.mean(coh[mask])))
                    else:
                        features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _plv_features(
        self, epoch: np.ndarray, sfreq: float
    ) -> np.ndarray:
        """
        Phase Locking Value (PLV) — Hilbert transform orqali kanal juftlari
        orasidagi faza sinxroniyasini o'lchaydi.

        PLV ∈ [0, 1]: 0 = hech qanday sinxronlik, 1 = to'liq sinxron.
        """
        from scipy.signal import hilbert

        n_channels = epoch.shape[0]
        # Hilbert transform → analytic signal → faza
        try:
            analytic = hilbert(epoch, axis=1)
            phases = np.angle(analytic)           # (C, T)
        except Exception:
            n_pairs = n_channels * (n_channels - 1) // 2
            return np.zeros(n_pairs, dtype=np.float32)

        features: List[float] = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = phases[i] - phases[j]
                plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
                features.append(plv)

        return np.array(features, dtype=np.float32)
