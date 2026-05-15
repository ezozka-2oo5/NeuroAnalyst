"""
ml_pipeline/features/temporal.py
====================================
EEG signallaridan vaqt domenidagi statistik xususiyatlarni ajratib olish.

Har bir kanal uchun:
  - Statistik momentlar (mean, std, skewness, kurtosis)
  - Amplitude xususiyatlari (peak-to-peak, RMS, min, max)
  - Hjorth parametrlari (Activity, Mobility, Complexity)
  - Approximate Entropy (ApEn)
  - Sample Entropy (SampEn)
  - Hurst eksponent (fraktal xususiyat)
  - Zero crossing rate
  - Line length
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
from scipy.stats import kurtosis, skew

from ..utils.config import FeatureConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TemporalFeatureExtractor:
    """
    Vaqt domenidagi EEG xususiyatlari.

    Qo'llanilishi::

        extractor = TemporalFeatureExtractor(cfg)
        features = extractor.extract(epoch, sfreq=256.0)
    """

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def extract(
        self, epoch: np.ndarray, sfreq: float
    ) -> np.ndarray:
        """
        Bitta epoch dan temporal featurelarni ajratib oladi.

        Args:
            epoch: (n_channels, n_timepoints) float32.
            sfreq: Hz.

        Returns:
            1D feature vector, float32.
        """
        all_features: List[np.ndarray] = []

        for ch_idx in range(epoch.shape[0]):
            ch_signal = epoch[ch_idx].astype(np.float64)
            ch_features = self._extract_channel_features(ch_signal, sfreq)
            all_features.append(ch_features)

        return np.concatenate(all_features).astype(np.float32)

    def extract_batch(
        self, epochs: np.ndarray, sfreq: float
    ) -> np.ndarray:
        """Batch feature extraction."""
        results = [self.extract(epochs[i], sfreq) for i in range(len(epochs))]
        return np.stack(results, axis=0).astype(np.float32)

    def feature_names(
        self, channel_names: Optional[List[str]] = None, n_channels: int = 19
    ) -> List[str]:
        """Feature nomlarini qaytaradi."""
        names: List[str] = []
        ch_labels = channel_names if channel_names else [f"Ch{i}" for i in range(n_channels)]
        base_names = [
            "mean", "std", "var", "skewness", "kurtosis",
            "min", "max", "peak2peak", "rms",
            "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
            "zero_crossing_rate", "line_length",
        ]
        if self.config.use_entropy:
            base_names += ["approx_entropy", "sample_entropy"]
        if self.config.use_hjorth:
            pass  # already included above
        for ch in ch_labels:
            for feat in base_names:
                names.append(f"{ch}_{feat}")
        return names

    # ──────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────

    def _extract_channel_features(
        self, signal: np.ndarray, sfreq: float
    ) -> np.ndarray:
        """Bitta kanal signalidan temporal featurelar."""
        features: List[float] = []

        # ── Statistik momentlar ────────────────────────────────
        features.append(float(np.mean(signal)))
        features.append(float(np.std(signal)))
        features.append(float(np.var(signal)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features.append(float(skew(signal)))
            features.append(float(kurtosis(signal)))

        # ── Amplitude ─────────────────────────────────────────
        features.append(float(np.min(signal)))
        features.append(float(np.max(signal)))
        features.append(float(np.max(signal) - np.min(signal)))           # Peak-to-peak
        features.append(float(np.sqrt(np.mean(signal ** 2))))             # RMS

        # ── Hjorth parametrlari ───────────────────────────────
        activity, mobility, complexity = self._hjorth(signal)
        features.append(float(activity))
        features.append(float(mobility))
        features.append(float(complexity))

        # ── Zero crossing rate ────────────────────────────────
        zcr = float(np.sum(np.diff(np.sign(signal)) != 0)) / len(signal)
        features.append(zcr)

        # ── Line length (EEG seizure detection da klassik feature) ─
        ll = float(np.sum(np.abs(np.diff(signal))))
        features.append(ll)

        # ── Entropy featurelari ───────────────────────────────
        if self.config.use_entropy:
            features.append(self._approximate_entropy(signal, m=2, r=0.2))
            features.append(self._sample_entropy(signal, m=2, r=0.2))

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _hjorth(signal: np.ndarray) -> tuple:
        """
        Hjorth parametrlarini hisoblaydi.

        - Activity: Signalning dispersiyasi (quvvat)
        - Mobility: O'rtacha chastota haqida ma'lumot
        - Complexity: Chastota tarkibining murakkabligi
        """
        activity = float(np.var(signal))

        d1 = np.diff(signal)
        var_d1 = float(np.var(d1)) if len(d1) > 0 else 0.0
        mobility = float(np.sqrt(var_d1 / (activity + 1e-12)))

        if len(d1) > 1:
            d2 = np.diff(d1)
            var_d2 = float(np.var(d2))
            mob_d1 = float(np.sqrt(var_d2 / (var_d1 + 1e-12)))
        else:
            mob_d1 = 0.0

        complexity = mob_d1 / (mobility + 1e-12)

        return activity, mobility, complexity

    @staticmethod
    def _approximate_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Approximate Entropy (ApEn) — signalning tartibsizlik darajasi.

        Kichik ApEn → regularroq signal (masalan, seizure paytida ko'pincha yuqori).

        Args:
            signal: 1D float array.
            m: Template uzunligi.
            r: Matching threshold (std fraksiyasi).
        """
        N = len(signal)
        if N < m + 2:
            return 0.0

        r_threshold = r * float(np.std(signal)) + 1e-12

        def _phi(m_len):
            x = np.array([signal[i:i + m_len] for i in range(N - m_len + 1)])
            count = np.sum(
                np.max(np.abs(x[:, np.newaxis] - x[np.newaxis, :]), axis=2) <= r_threshold,
                axis=0,
            )
            count = np.maximum(count - 1, 0)   # o'zini sanash yo'q
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                log_vals = np.where(count > 0, np.log(count / (N - m_len)), 0.0)
            return float(np.mean(log_vals))

        try:
            return _phi(m) - _phi(m + 1)
        except Exception:
            return 0.0

    @staticmethod
    def _sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Sample Entropy (SampEn) — ApEn ning yaxshilangan versiyasi.

        O'z-o'zini sanash yo'q, shuning uchun ApEn ga nisbatan aniqliqli.
        """
        N = len(signal)
        if N < m + 2:
            return 0.0

        r_threshold = r * float(np.std(signal)) + 1e-12

        def _count_matches(m_len):
            templates = np.array([signal[i:i + m_len] for i in range(N - m_len)])
            total = 0
            for i in range(len(templates)):
                diffs = np.max(np.abs(templates - templates[i]), axis=1)
                total += np.sum(diffs <= r_threshold) - 1  # o'zini sanama
            return total

        try:
            B = _count_matches(m)
            A = _count_matches(m + 1)
            if B == 0:
                return 0.0
            return float(-np.log(A / B)) if A > 0 else 0.0
        except Exception:
            return 0.0
