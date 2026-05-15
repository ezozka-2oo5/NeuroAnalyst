"""
ml_pipeline/preprocessing/epoching.py
========================================
Uzun EEG yozuvini kichik o'qitish bo'laklariga (epoch) bo'luvchi modul.

Sliding window usuli bilan overlapping epochlar yaratiladi —
bu data augmentatsiyaning eng oddiy shakli.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.config import PreprocessingConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EpochMaker:
    """
    Uzun EEG signalini bir xil o'lchamli epoch segmentlarga bo'ladi.

    Har bir epoch (n_channels × n_timepoints) shaklidagi numpy array.

    Qo'llanilishi::

        maker = EpochMaker(cfg)

        # Event-based epochlar (seizure/normal)
        epochs, labels = maker.epoch_with_events(
            data, sfreq, events=[(512, 1), (1024, 0)], window_sec=4.0
        )

        # Sliding window epochlar (tasnifsiz)
        epochs = maker.sliding_window(data, sfreq, window_sec=4.0, overlap_sec=2.0)
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    # ──────────────────────────────────────────────────────────────
    # Public: Sliding window
    # ──────────────────────────────────────────────────────────────

    def sliding_window(
        self,
        data: np.ndarray,
        sfreq: float,
        window_sec: Optional[float] = None,
        overlap_sec: Optional[float] = None,
        reject_artifacts: bool = True,
    ) -> List[np.ndarray]:
        """
        Signalni sliding window bilan epochlaydi.

        Args:
            data: (n_channels, n_samples) µV.
            sfreq: Hz.
            window_sec: Epoch uzunligi soniyada. None bo'lsa config dan.
            overlap_sec: Overlap uzunligi soniyada. None bo'lsa config dan.
            reject_artifacts: True bo'lsa yomon epochlar tashlanadi.

        Returns:
            Epoch ro'yxati: har biri (n_channels, n_timepoints) shape.
        """
        win_sec = window_sec or self.config.epoch_duration_sec
        ovlp_sec = overlap_sec or self.config.epoch_overlap_sec

        win_samples = int(win_sec * sfreq)
        step_samples = max(1, int((win_sec - ovlp_sec) * sfreq))

        n_samples = data.shape[1]
        epochs: List[np.ndarray] = []
        rejected = 0

        start = 0
        while start + win_samples <= n_samples:
            end = start + win_samples
            epoch = data[:, start:end].copy().astype(np.float32)

            if reject_artifacts and self._is_artifact(epoch):
                rejected += 1
            else:
                epochs.append(epoch)

            start += step_samples

        logger.debug(
            f"Epochlash: {len(epochs)} ta qabul qilindi, {rejected} ta rad etildi. "
            f"(window={win_sec}s, overlap={ovlp_sec}s)"
        )
        return epochs

    # ──────────────────────────────────────────────────────────────
    # Public: Event-based epochlar
    # ──────────────────────────────────────────────────────────────

    def epoch_with_events(
        self,
        data: np.ndarray,
        sfreq: float,
        events: List[Tuple[int, int, int]],   # (onset_sample, label, duration_samples)
        window_sec: Optional[float] = None,
        pre_event_sec: float = 0.0,
        post_event_sec: Optional[float] = None,
        reject_artifacts: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Event (annotation) vaqtlari asosida epochlar yaratadi.

        Har bir event uchun [onset - pre_event_sec, onset + post_event_sec]
        oralig'idagi signal olinadi.

        Args:
            data: (n_channels, n_samples) µV.
            sfreq: Hz.
            events: [(onset_sample, label, duration_samples), ...].
            window_sec: Agar event davomiyligini ishlatmasdan bir xil oyna kerak bo'lsa.
            pre_event_sec: Event boshlanishidan oldin olingan vaqt (soniyada).
            post_event_sec: Event boshlanishidan keyin olingan vaqt (soniyada).
                            None bo'lsa window_sec ishlatiladi.
            reject_artifacts: Yomon epochlarni tashlasin.

        Returns:
            (X, y) tuple:
              - X: shape (n_epochs, n_channels, n_timepoints) float32.
              - y: shape (n_epochs,) int64.
        """
        win_sec = window_sec or self.config.epoch_duration_sec
        pre_samples = int(pre_event_sec * sfreq)
        post_samples = int((post_event_sec or win_sec) * sfreq)
        epoch_len = pre_samples + post_samples
        n_samples = data.shape[1]

        epoch_list: List[np.ndarray] = []
        label_list: List[int] = []
        rejected = 0

        for onset_sample, label, duration_samples in events:
            start = onset_sample - pre_samples
            end = start + epoch_len

            # Chegaralardan chiqmasin
            if start < 0 or end > n_samples:
                logger.debug(f"Epoch chegaradan chiqdi: [{start}, {end}]. O'tkazildi.")
                continue

            epoch = data[:, start:end].copy().astype(np.float32)

            if reject_artifacts and self._is_artifact(epoch):
                rejected += 1
                continue

            epoch_list.append(epoch)
            label_list.append(label)

        if not epoch_list:
            logger.warning("Hech qanday epoch yaratilmadi. Events yoki chegaralarni tekshiring.")
            return np.empty((0, data.shape[0], epoch_len), dtype=np.float32), np.empty(0, dtype=np.int64)

        X = np.stack(epoch_list, axis=0)         # (N, C, T)
        y = np.array(label_list, dtype=np.int64)

        logger.info(
            f"Event-based epochlar: {len(X)} ta yaratildi, {rejected} ta rad etildi. "
            f"Label taqsimoti: { {int(lb): int(np.sum(y == lb)) for lb in np.unique(y)} }"
        )
        return X, y

    def epoch_with_seizure_annotations(
        self,
        data: np.ndarray,
        sfreq: float,
        seizure_intervals: List[Tuple[float, float]],  # [(start_sec, end_sec), ...]
        window_sec: Optional[float] = None,
        overlap_sec: Optional[float] = None,
        normal_ratio: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CHB-MIT kabi seizure interval annotatsiyalari asosida epochlar yaratadi.

        Agar seizure_intervals bo'lsa: seizure epochlar + normal epochlar.
        normal_ratio: Har 1 seizure epoch uchun nechta normal epoch olinsin.
        Bu class imbalance bilan kurashishga yordam beradi.

        Args:
            data: (n_channels, n_samples) µV.
            sfreq: Hz.
            seizure_intervals: [(boshlash_soniya, tugash_soniya), ...].
            window_sec: Epoch oyna uzunligi.
            overlap_sec: Overlap.
            normal_ratio: Normal : Seizure nisbati.

        Returns:
            (X, y) tuple.
        """
        win_sec = window_sec or self.config.epoch_duration_sec
        ovlp_sec = overlap_sec or self.config.epoch_overlap_sec
        win_samples = int(win_sec * sfreq)

        # Seizure mask (boolean)
        n_samples = data.shape[1]
        seizure_mask = np.zeros(n_samples, dtype=bool)
        for start_sec, end_sec in seizure_intervals:
            s = max(0, int(start_sec * sfreq))
            e = min(n_samples, int(end_sec * sfreq))
            seizure_mask[s:e] = True

        seizure_epochs: List[np.ndarray] = []
        normal_epochs: List[np.ndarray] = []

        all_sliding = self.sliding_window(data, sfreq, win_sec, ovlp_sec, reject_artifacts=True)
        step_samples = max(1, int((win_sec - ovlp_sec) * sfreq))

        for i, epoch in enumerate(all_sliding):
            start = i * step_samples
            end = start + win_samples
            # Epoch ichida seizure bormi? 50% dan ko'p bo'lsa seizure deb belgilaymiz
            overlap = np.mean(seizure_mask[start:end]) if end <= n_samples else 0.0
            if overlap >= 0.5:
                seizure_epochs.append(epoch)
            else:
                normal_epochs.append(epoch)

        # Balans: normallarni cheklaymiz
        n_seizure = len(seizure_epochs)
        n_normal_needed = min(len(normal_epochs), int(n_seizure * normal_ratio))

        # Normal epochlarni tasodifiy tanlaymiz
        rng = np.random.default_rng(42)
        if normal_epochs:
            normal_indices = rng.choice(len(normal_epochs), size=n_normal_needed, replace=False)
            normal_epochs = [normal_epochs[i] for i in normal_indices]

        epoch_list = seizure_epochs + normal_epochs
        label_list = [1] * n_seizure + [0] * len(normal_epochs)

        if not epoch_list:
            return np.empty((0, data.shape[0], win_samples), dtype=np.float32), np.empty(0, dtype=np.int64)

        X = np.stack(epoch_list, axis=0)
        y = np.array(label_list, dtype=np.int64)

        # Aralashtiramiz
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]

        logger.info(
            f"Seizure epochlash: {n_seizure} seizure + {len(normal_epochs)} normal = {len(X)} jami."
        )
        return X, y

    # ──────────────────────────────────────────────────────────────
    # Baseline correction
    # ──────────────────────────────────────────────────────────────

    def apply_baseline_correction(
        self,
        epochs: np.ndarray,
        sfreq: float,
        baseline_window: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Epochlarga baseline correction qo'llaydi.

        Baseline window ichidagi o'rtacha qiymat har bir kanaldan ayiriladi.

        Args:
            epochs: (N, n_channels, n_timepoints) float32.
            sfreq: Hz.
            baseline_window: (start_sec, end_sec) soniyada.
                             None bo'lsa config dan olinadi.

        Returns:
            Baseline corrected epochs.
        """
        if not self.config.baseline_correction:
            return epochs

        bw = baseline_window or self.config.baseline_window
        start_sample = max(0, int(bw[0] * sfreq))
        end_sample = int(bw[1] * sfreq)

        if end_sample <= start_sample:
            logger.warning("Baseline oyna noto'g'ri. Correction qo'llanilmaydi.")
            return epochs

        baseline_mean = np.mean(epochs[:, :, start_sample:end_sample], axis=2, keepdims=True)
        corrected = epochs - baseline_mean
        logger.debug(f"Baseline correction: [{bw[0]:.2f}s, {bw[1]:.2f}s]")
        return corrected.astype(np.float32)

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _is_artifact(self, epoch: np.ndarray) -> bool:
        """
        Bitta epoch artefaktli ekanligini tekshiradi.

        Mezonlar:
        - Maksimal amplitude threshold dan oshsa.
        - Kanallardan biri butunlay flat bo'lsa.
        """
        threshold = self.config.epoch_reject_threshold_uv
        flat_thr = self.config.flat_channel_threshold_uv

        # Amplitude threshold
        if np.any(np.abs(epoch) > threshold):
            return True

        # Flat channel
        if np.any(np.std(epoch, axis=1) < flat_thr):
            return True

        return False
