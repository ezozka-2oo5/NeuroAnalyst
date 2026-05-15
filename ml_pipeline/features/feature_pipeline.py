"""
ml_pipeline/features/feature_pipeline.py
==========================================
Barcha feature extractorlarni birlashtirib, scikit-learn uchun
feature matrix (X) va nom ro'yxatini yaratuvchi klass.

Shuningdek PCA dimensionality reduction ni amalga oshiradi.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from .spectral import SpectralFeatureExtractor
from .temporal import TemporalFeatureExtractor
from .connectivity import ConnectivityFeatureExtractor
from ..utils.config import FeatureConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """
    Barcha featurelarni birlashtiruvchi pipeline.

    Classical ML modellar uchun epoch to'plamidan feature matrix yaratadi.
    Deep learning modellari bu pipelineni ishlatmaydi — ular to'g'ridan-to'g'ri
    raw epoch larni qabul qiladi.

    Qo'llanilishi::

        fp = FeaturePipeline(cfg, sfreq=256.0)
        X_feats = fp.fit_transform(X_epochs_train)   # (N, n_features)
        X_test_feats = fp.transform(X_epochs_test)   # (N, n_features)
        fp.save("feature_pipeline.pkl")
    """

    def __init__(
        self,
        config: FeatureConfig,
        sfreq: float = 256.0,
        channel_names: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self.sfreq = sfreq
        self.channel_names = channel_names

        self.spectral_extractor = SpectralFeatureExtractor(config) if config.use_spectral else None
        self.temporal_extractor = TemporalFeatureExtractor(config) if config.use_temporal else None
        self.connectivity_extractor = ConnectivityFeatureExtractor(config) if config.use_connectivity else None

        self.scaler = RobustScaler()
        self.pca: Optional[PCA] = None
        self._fitted = False

    def extract_features(self, epochs: np.ndarray) -> np.ndarray:
        """
        Epochlar dan xom (unscaled) featurelarni ajratib oladi.

        Args:
            epochs: (N, n_channels, n_timepoints) float32.

        Returns:
            (N, n_features) float32.
        """
        feature_parts: List[np.ndarray] = []

        if self.spectral_extractor is not None:
            logger.info("Spectral featurelar hisoblanmoqda...")
            sp_feats = self.spectral_extractor.extract_batch(epochs, self.sfreq)
            feature_parts.append(sp_feats)
            logger.info(f"  Spectral: {sp_feats.shape[1]} feature")

        if self.temporal_extractor is not None:
            logger.info("Temporal featurelar hisoblanmoqda...")
            tm_feats = self.temporal_extractor.extract_batch(epochs, self.sfreq)
            feature_parts.append(tm_feats)
            logger.info(f"  Temporal: {tm_feats.shape[1]} feature")

        if self.connectivity_extractor is not None:
            logger.info("Connectivity featurelar hisoblanmoqda...")
            cn_feats = self.connectivity_extractor.extract_batch(epochs, self.sfreq)
            feature_parts.append(cn_feats)
            logger.info(f"  Connectivity: {cn_feats.shape[1]} feature")

        if not feature_parts:
            raise ValueError("Hech qanday feature extractor yoqilmagan.")

        X = np.concatenate(feature_parts, axis=1)
        # NaN va Inf qiymatlarni tozalaymiz
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X.astype(np.float32)

    def fit_transform(self, epochs: np.ndarray) -> np.ndarray:
        """
        Feature ajratish + scaler fit + (ixtiyoriy) PCA.
        Faqat train to'plami uchun.

        Args:
            epochs: (N, n_channels, n_timepoints) float32.

        Returns:
            (N, n_features) float32.
        """
        X = self.extract_features(epochs)

        # Scaler fit va transform
        X = self.scaler.fit_transform(X).astype(np.float32)
        logger.info(f"RobustScaler fit qilindi. Feature soni: {X.shape[1]}")

        # PCA
        if self.config.apply_pca:
            n_components = min(
                int(self.config.pca_variance_ratio * X.shape[1]),
                X.shape[0] - 1,
                X.shape[1],
            )
            self.pca = PCA(n_components=n_components, random_state=42)
            X = self.pca.fit_transform(X).astype(np.float32)
            explained = float(np.sum(self.pca.explained_variance_ratio_)) * 100
            logger.info(
                f"PCA qo'llandi: {X.shape[1]} komponent, "
                f"{explained:.1f}% dispersiya saqlanmoqda."
            )

        self._fitted = True
        return X

    def transform(self, epochs: np.ndarray) -> np.ndarray:
        """
        Hisoblangan scaler/PCA bilan yangi epochlarni transform qiladi.

        Args:
            epochs: (N, n_channels, n_timepoints) float32.

        Returns:
            (N, n_features) float32.
        """
        if not self._fitted:
            raise RuntimeError("fit_transform() ni avval chaqiring.")

        X = self.extract_features(epochs)
        X = self.scaler.transform(X).astype(np.float32)

        if self.pca is not None:
            X = self.pca.transform(X).astype(np.float32)

        return X

    def get_feature_names(self) -> List[str]:
        """Barcha feature nomlarini qaytaradi."""
        n_ch = self.channel_names.__len__() if self.channel_names else 19
        names: List[str] = []
        if self.spectral_extractor:
            names.extend(self.spectral_extractor.feature_names(self.channel_names, n_ch))
        if self.temporal_extractor:
            names.extend(self.temporal_extractor.feature_names(self.channel_names, n_ch))
        return names

    def save(self, path: str | Path) -> None:
        """Pipeline holatini diskka saqlaydi."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "config": self.config,
                    "sfreq": self.sfreq,
                    "channel_names": self.channel_names,
                    "scaler": self.scaler,
                    "pca": self.pca,
                    "fitted": self._fitted,
                },
                fh,
            )
        logger.info(f"FeaturePipeline saqlandi: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        """Diskdan pipeline holatini yuklaydi."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(
            config=state["config"],
            sfreq=state["sfreq"],
            channel_names=state["channel_names"],
        )
        obj.scaler = state["scaler"]
        obj.pca = state["pca"]
        obj._fitted = state["fitted"]
        logger.info(f"FeaturePipeline yuklandi: {path}")
        return obj
