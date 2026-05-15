"""
ml_pipeline/preprocessing/pipeline.py
========================================
Butun preprocessing zanjirini boshqaruvchi asosiy Pipeline klassi.

Bu klass filtrlash → artefakt olib tashlash → resampling →
kanal tanlash → epochlash → normalizatsiya qadamlarini ketma-ket bajaradi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .artifact_removal import ArtifactRemover
from .epoching import EpochMaker
from .filters import EEGFilter
from .normalization import SignalNormalizer
from ..utils.config import DataConfig, PreprocessingConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """
    To'liq EEG preprocessing zanjiri.

    Foydalanish::

        cfg_data = DataConfig()
        cfg_prep = PreprocessingConfig()
        pipeline = PreprocessingPipeline(cfg_data, cfg_prep)

        # Bitta yozuv uchun
        epochs, quality = pipeline.process_recording(recording_dict)

        # Butun dataset uchun
        X, y = pipeline.process_dataset(recordings_list)
    """

    def __init__(
        self,
        data_config: DataConfig,
        prep_config: PreprocessingConfig,
    ) -> None:
        self.data_config = data_config
        self.prep_config = prep_config

        self.filter = EEGFilter(prep_config)
        self.artifact_remover = ArtifactRemover(prep_config)
        self.epoch_maker = EpochMaker(prep_config)
        self.normalizer = SignalNormalizer(method="robust")

    def process_recording(
        self,
        recording: dict,
        seizure_intervals: Optional[List[Tuple[float, float]]] = None,
        events: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Bitta yozuvni to'liq processing qiladi.

        Args:
            recording: BIDSParser.load_edf() natijasi dict:
                       {"data", "sfreq", "channel_names", "label", ...}
            seizure_intervals: CHB-MIT uchun [(start_sec, end_sec), ...].
            events: Event-based epochlash uchun [(onset_sample, label, dur), ...].

        Returns:
            (X, y, quality_metrics) tuple:
              - X: (N, n_channels, n_timepoints) float32.
              - y: (N,) int64 label array.
              - quality_metrics: Signal sifati ko'rsatkichlari dict.
        """
        data = recording["data"].astype(np.float32)        # (C, T)
        sfreq = float(recording["sfreq"])
        channel_names = recording.get("channel_names", [])
        recording_label = recording.get("label", -1)

        logger.info(
            f"Processing: {recording.get('source_name', '?')} | "
            f"{data.shape[0]} kanal | {sfreq:.0f} Hz | {data.shape[1]/sfreq:.1f}s"
        )

        # ── 1. Kanal tanlash ──────────────────────────────────────
        data, channel_names = self._select_channels(data, channel_names)
        if data.shape[0] == 0:
            logger.warning("Kerakli kanallar topilmadi. Processing to'xtatildi.")
            return np.empty(0), np.empty(0), {}

        # ── 2. Resampling ─────────────────────────────────────────
        target_sfreq = self.data_config.target_sfreq
        if abs(sfreq - target_sfreq) > 1.0:
            data, sfreq = self._resample(data, sfreq, target_sfreq)

        # ── 3. Filtrlash ─────────────────────────────────────────
        data = self.filter.apply_all(data, sfreq)

        # ── 4. Yomon kanallarni aniqlash va interpolatsiya ────────
        bad_idx, bad_names = self.artifact_remover.detect_bad_channels(
            data, sfreq, channel_names
        )
        if bad_idx:
            data = self.artifact_remover.interpolate_bad_channels(
                data, bad_idx, channel_names
            )

        # ── 5. Sifat ko'rsatkichlari ──────────────────────────────
        quality_metrics = self.artifact_remover.compute_artifact_ratio(data, sfreq)
        quality_metrics["bad_channels"] = bad_names
        quality_metrics["n_channels"] = data.shape[0]
        quality_metrics["duration_sec"] = data.shape[1] / sfreq

        # ── 6. Epochlash ─────────────────────────────────────────
        if seizure_intervals is not None:
            # CHB-MIT style: seizure intervallari berilgan
            X, y = self.epoch_maker.epoch_with_seizure_annotations(
                data, sfreq, seizure_intervals
            )
        elif events is not None:
            # Event-based epochlash
            X, y = self.epoch_maker.epoch_with_events(data, sfreq, events)
        else:
            # Sliding window — label butun yozuvdan
            epoch_list = self.epoch_maker.sliding_window(data, sfreq)
            if not epoch_list:
                return np.empty(0), np.empty(0), quality_metrics
            X = np.stack(epoch_list, axis=0)
            y = np.full(len(X), fill_value=recording_label, dtype=np.int64)

        if len(X) == 0:
            return np.empty(0), np.empty(0), quality_metrics

        # ── 7. Baseline correction ────────────────────────────────
        X = self.epoch_maker.apply_baseline_correction(X, sfreq)

        quality_metrics["n_epochs_created"] = len(X)
        logger.info(f"  → {len(X)} epoch yaratildi. Sifat: {quality_metrics['signal_quality_score']:.1f}/100")
        return X, y, quality_metrics

    def process_dataset(
        self,
        recordings: List[dict],
        seizure_map: Optional[Dict[str, List[Tuple[float, float]]]] = None,
        fit_normalizer: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Butun dataset ni processing qilib, X va y ni qaytaradi.

        Args:
            recordings: BIDSParser.load_dataset() natijasi ro'yxat.
            seizure_map: {fayl_nomi: [(start_sec, end_sec), ...]} CHB-MIT uchun.
            fit_normalizer: True bo'lsa, normalizer ushbu datasetdan fit qilinadi.

        Returns:
            (X, y) tuple:
              - X: (N_total, n_channels, n_timepoints) float32.
              - y: (N_total,) int64.
        """
        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        failed = 0

        for idx, recording in enumerate(recordings):
            name = recording.get("source_name", f"recording_{idx}")
            logger.info(f"[{idx + 1}/{len(recordings)}] {name}")

            # Seizure intervallari
            intervals = None
            if seizure_map:
                intervals = seizure_map.get(name) or seizure_map.get(
                    recording.get("source_path", "")
                )

            try:
                X, y, quality = self.process_recording(recording, seizure_intervals=intervals)
            except Exception as exc:
                logger.warning(f"  Xato ({name}): {exc}. O'tkazildi.")
                failed += 1
                continue

            if len(X) == 0:
                logger.warning(f"  Bo'sh natija: {name}. O'tkazildi.")
                continue

            all_X.append(X)
            all_y.append(y)

        if not all_X:
            raise RuntimeError("Hech qanday yozuv muvaffaqiyatli qayta ishlanmadi.")

        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)

        logger.info(
            f"Dataset processing tugadi: "
            f"{len(X_all)} epoch | {failed} xato | "
            f"Shape: {X_all.shape} | "
            f"Labels: { {int(l): int(np.sum(y_all == l)) for l in np.unique(y_all)} }"
        )

        # Normalizatsiya
        if fit_normalizer:
            X_all = self.normalizer.fit_transform(X_all)
        else:
            X_all = self.normalizer.transform(X_all)

        return X_all, y_all

    def save_scaler(self, path: str | Path) -> None:
        """Normalizer holatini saqlaydi."""
        self.normalizer.save(path)

    def load_scaler(self, path: str | Path) -> None:
        """Saqlangan normalizer holatini yuklaydi."""
        self.normalizer = SignalNormalizer.load(path)

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _select_channels(
        self, data: np.ndarray, channel_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Konfiguratsiyada ko'rsatilgan kanallarni tanlab oladi.
        Agar kanallar ro'yxati bo'lmasa, hamma kanallar qaytariladi.
        """
        target_channels = self.data_config.channels_to_use
        if target_channels is None or not channel_names:
            return data, channel_names

        selected_indices: List[int] = []
        selected_names: List[str] = []

        # Case-insensitive matching
        ch_lower = {ch.lower(): idx for idx, ch in enumerate(channel_names)}
        for target in target_channels:
            idx = ch_lower.get(target.lower())
            if idx is not None:
                selected_indices.append(idx)
                selected_names.append(channel_names[idx])
            else:
                logger.debug(f"Kanal topilmadi: '{target}'")

        if not selected_indices:
            logger.warning(
                "Hech qanday kerakli kanal topilmadi. Barcha kanallar qaytariladi."
            )
            return data, channel_names

        return data[selected_indices, :], selected_names

    @staticmethod
    def _resample(
        data: np.ndarray, sfreq: float, target_sfreq: float
    ) -> Tuple[np.ndarray, float]:
        """
        Signalni scipy resampling orqali boshqa chastotaga o'tkazadi.

        MNE mavjud bo'lsa, undan foydalanadi (aniqroq anti-aliasing).
        """
        try:
            import mne
            from mne.filter import resample

            data_resampled = resample(data.astype(np.float64), up=target_sfreq, down=sfreq, npad="auto", axis=1)
            logger.info(f"MNE resampling: {sfreq:.1f} → {target_sfreq:.1f} Hz")
            return data_resampled.astype(np.float32), target_sfreq

        except ImportError:
            pass

        # scipy fallback
        from scipy.signal import resample as scipy_resample

        n_original = data.shape[1]
        ratio = target_sfreq / sfreq
        n_target = int(n_original * ratio)
        data_resampled = scipy_resample(data, n_target, axis=1)
        logger.info(f"scipy resampling: {sfreq:.1f} → {target_sfreq:.1f} Hz")
        return data_resampled.astype(np.float32), target_sfreq
