"""
ml_pipeline/features/spectral.py
====================================
EEG signallaridan chastota domenidagi xususiyatlarni ajratib olish.

Har bir kanal uchun quyidagi featurelar hisoblanadi:
  - Band quvvatlari (delta, theta, alpha, beta, gamma)
  - Nisbiy band quvvatlari (relatif power)
  - Dominant chastota (peak frequency)
  - Spectral entropy
  - Alpha/Theta nisbati va boshqa klinik indekslar
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch

from ..utils.config import FeatureConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Numpy 2.0 compatibility
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# EEG ritmlar (Hz)
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}


class SpectralFeatureExtractor:
    """
    EEG epochdan chastota domenidagi xususiyatlarni ajratib oladi.

    Har bir epoch uchun barcha kanallar bo'yicha featuralar hisoblaning va
    bitta yassi vektor (1D) sifatida qaytariladi.

    Qo'llanilishi::

        cfg = FeatureConfig()
        extractor = SpectralFeatureExtractor(cfg)
        features = extractor.extract(epoch, sfreq=256.0)
        # features.shape → (n_channels × n_features_per_channel,)
    """

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.bands = config.bands or BANDS

    def extract(
        self, epoch: np.ndarray, sfreq: float
    ) -> np.ndarray:
        """
        Bitta epoch dan spectral featurelarni ajratib oladi.

        Args:
            epoch: (n_channels, n_timepoints) float32, µV.
            sfreq: Sampling chastotasi Hz.

        Returns:
            1D feature vector, float32.
        """
        n_channels = epoch.shape[0]
        all_features: List[np.ndarray] = []

        # Welch PSD — barcha kanallar uchun bir vaqtda
        freqs, psd = welch(
            epoch,
            fs=sfreq,
            nperseg=min(self.config.welch_nperseg, epoch.shape[1]),
            noverlap=min(self.config.welch_noverlap, epoch.shape[1] // 2),
            axis=1,
        )
        # psd shape: (n_channels, n_freqs)

        for ch_idx in range(n_channels):
            ch_psd = psd[ch_idx]
            ch_features = self._extract_channel_features(ch_psd, freqs)
            all_features.append(ch_features)

        return np.concatenate(all_features).astype(np.float32)

    def extract_batch(
        self, epochs: np.ndarray, sfreq: float
    ) -> np.ndarray:
        """
        Bir nechta epoch uchun batch-da feature extraction.

        Args:
            epochs: (N, n_channels, n_timepoints) float32.
            sfreq: Hz.

        Returns:
            (N, n_features) float32 feature matrix.
        """
        results = [self.extract(epochs[i], sfreq) for i in range(len(epochs))]
        return np.stack(results, axis=0).astype(np.float32)

    def feature_names(self, channel_names: Optional[List[str]] = None, n_channels: int = 19) -> List[str]:
        """Feature nomlarining ro'yxatini qaytaradi (scikit-learn uchun)."""
        names: List[str] = []
        ch_labels = channel_names if channel_names else [f"Ch{i}" for i in range(n_channels)]
        for ch in ch_labels:
            for band_name in self.bands:
                names.append(f"{ch}_pwr_{band_name}")
                names.append(f"{ch}_relpwr_{band_name}")
            names.append(f"{ch}_peak_freq")
            names.append(f"{ch}_spectral_entropy")
            names.append(f"{ch}_spectral_edge_50")
            names.append(f"{ch}_spectral_edge_90")
            names.append(f"{ch}_alpha_theta_ratio")
        return names

    # ──────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────

    def _extract_channel_features(
        self, psd: np.ndarray, freqs: np.ndarray
    ) -> np.ndarray:
        """Bitta kanal PSD dan featurelar."""
        features: List[float] = []
        total_power = float(_trapz(psd, freqs)) or 1e-12

        band_powers: Dict[str, float] = {}
        for band_name, (low, high) in self.bands.items():
            mask = (freqs >= low) & (freqs < high)
            bp = float(_trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
            band_powers[band_name] = bp

        # Band powers va relative powers
        for band_name in self.bands:
            bp = band_powers[band_name]
            features.append(bp)
            features.append(bp / total_power)

        # Dominant (peak) chastota
        peak_idx = np.argmax(psd)
        features.append(float(freqs[peak_idx]))

        # Spectral entropy (Shannon entropy PSD bo'yicha)
        psd_norm = psd / (total_power + 1e-12)
        psd_norm = np.clip(psd_norm, 1e-12, None)
        spectral_entropy = float(-np.sum(psd_norm * np.log(psd_norm)))
        features.append(spectral_entropy)

        # Spectral edge frequency (50% va 90% kumulyativ quvvat)
        cumulative = np.cumsum(psd)
        cumulative /= cumulative[-1] + 1e-12
        sef50_idx = np.searchsorted(cumulative, 0.50)
        sef90_idx = np.searchsorted(cumulative, 0.90)
        features.append(float(freqs[min(sef50_idx, len(freqs) - 1)]))
        features.append(float(freqs[min(sef90_idx, len(freqs) - 1)]))

        # Alpha/Theta nisbati (klinik muhim: uyg'oqlik darajasi)
        alpha = band_powers.get("alpha", 0.0)
        theta = band_powers.get("theta", 1e-12)
        features.append(alpha / (theta + 1e-12))

        return np.array(features, dtype=np.float32)
