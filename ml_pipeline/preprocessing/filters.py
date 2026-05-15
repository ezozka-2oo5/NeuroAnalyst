"""
ml_pipeline/preprocessing/filters.py
======================================
EEG signal filtrlash moduli.

Bandpass, notch va notch harmonik filtrlarni scipy.signal orqali
amalga oshiradi. Barcha operatsiyalar axis=1 bo'yicha ishlaydi
(shape: n_channels × n_samples).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from ..utils.config import PreprocessingConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EEGFilter:
    """
    EEG signallariga filtrlarni qo'llaydi.

    Qo'llanilishi::

        cfg = PreprocessingConfig()
        filt = EEGFilter(cfg)
        clean_data = filt.apply_all(raw_data, sfreq=256.0)
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def apply_all(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Barcha filtrlarni ketma-ket qo'llaydi:
          1. DC offset olib tashlash (de-mean)
          2. Bandpass filter
          3. Notch filter (asosiy va 2x harmonik)

        Args:
            data: shape (n_channels, n_samples), µV.
            sfreq: Sampling chastotasi Hz.

        Returns:
            Filtrlangan signal, xuddi shu shape.
        """
        # 1) DC offset (mean subtraction) per kanal
        data = self.remove_dc_offset(data)

        # 2) Bandpass
        data = self.bandpass(
            data, sfreq,
            low_cut=self.config.bandpass_low,
            high_cut=self.config.bandpass_high,
            order=self.config.bandpass_order,
        )

        # 3) Notch asosiy (50/60 Hz)
        if self.config.notch_hz > 0:
            data = self.notch(
                data, sfreq,
                freq_hz=self.config.notch_hz,
                q_factor=self.config.notch_q,
            )
            # 100/120 Hz harmonik
            harmonic = self.config.notch_hz * 2
            if harmonic < sfreq / 2:
                data = self.notch(data, sfreq, freq_hz=harmonic, q_factor=self.config.notch_q)

        return data

    @staticmethod
    def remove_dc_offset(data: np.ndarray) -> np.ndarray:
        """
        Har bir kanaldan o'rtacha qiymatni olib tashlaydi (DC offset removal).

        Args:
            data: shape (n_channels, n_samples).

        Returns:
            De-meaned signal.
        """
        return data - np.mean(data, axis=1, keepdims=True)

    @staticmethod
    def bandpass(
        data: np.ndarray,
        sfreq: float,
        low_cut: float = 0.5,
        high_cut: float = 45.0,
        order: int = 5,
        use_sos: bool = True,
    ) -> np.ndarray:
        """
        Zero-phase Butterworth bandpass filtr.

        ``filtfilt`` ikki tomonga ishlaydi, shuning uchun faza siljishi
        nolga teng bo'ladi — bu EEG tahlilida muhim.

        Args:
            data: (n_channels, n_samples) µV.
            sfreq: Sampling chastotasi Hz.
            low_cut: Pastki chegara Hz.
            high_cut: Yuqori chegara Hz.
            order: Butterworth tartib soni (4-6 optimal).
            use_sos: True bo'lsa ikkinchi tartibli bo'limlar (SOS) ishlatiladi.
                     Bu raqamli barqarorlik uchun yaxshiroq (yuqori order'da muhim).

        Returns:
            Filtrlangan signal.
        """
        nyq = sfreq / 2.0
        low = max(0.01, low_cut)
        high = min(high_cut, nyq - 0.5)

        if low >= high:
            logger.warning(
                f"Bandpass chegaralari noto'g'ri: [{low}, {high}]. Filter qo'llanilmaydi."
            )
            return data.copy()

        if use_sos:
            sos = butter(order, [low, high], btype="band", fs=sfreq, output="sos")
            return sosfiltfilt(sos, data, axis=1).astype(np.float32)
        else:
            b, a = butter(order, [low, high], btype="band", fs=sfreq)
            return filtfilt(b, a, data, axis=1).astype(np.float32)

    @staticmethod
    def notch(
        data: np.ndarray,
        sfreq: float,
        freq_hz: float = 50.0,
        q_factor: float = 30.0,
    ) -> np.ndarray:
        """
        Zero-phase notch (band-stop) filtr.

        Faqat bitta tor chastotani kesadi (masalan, 50 Hz elektr tarmog'i).

        Args:
            data: (n_channels, n_samples) µV.
            sfreq: Sampling chastotasi Hz.
            freq_hz: Kesilishi kerak bo'lgan chastota.
            q_factor: Notch filtri eni (kichikroq → kengroq kesiladi).

        Returns:
            Filtrlangan signal.
        """
        nyq = sfreq / 2.0
        if freq_hz <= 0 or freq_hz >= nyq:
            return data.copy()

        b, a = iirnotch(freq_hz, q_factor, fs=sfreq)
        return filtfilt(b, a, data, axis=1).astype(np.float32)

    @staticmethod
    def highpass(
        data: np.ndarray,
        sfreq: float,
        cut_freq: float = 0.1,
        order: int = 4,
    ) -> np.ndarray:
        """
        Highpass filtr — past chastotali drift va DC o'zgarishlarni kesadi.

        Args:
            data: (n_channels, n_samples) µV.
            sfreq: Sampling chastotasi Hz.
            cut_freq: Kesilishi kerak bo'lgan eng past chastota Hz.
            order: Butterworth tartibi.

        Returns:
            Filtrlangan signal.
        """
        nyq = sfreq / 2.0
        cut = min(cut_freq, nyq - 0.1)
        sos = butter(order, cut, btype="high", fs=sfreq, output="sos")
        return sosfiltfilt(sos, data, axis=1).astype(np.float32)

    @staticmethod
    def lowpass(
        data: np.ndarray,
        sfreq: float,
        cut_freq: float = 45.0,
        order: int = 4,
    ) -> np.ndarray:
        """
        Lowpass filtr — yuqori chastotali EMG shovqinini kesadi.

        Args:
            data: (n_channels, n_samples) µV.
            sfreq: Sampling chastotasi Hz.
            cut_freq: Kesilishi kerak bo'lgan eng yuqori chastota Hz.
            order: Butterworth tartibi.

        Returns:
            Filtrlangan signal.
        """
        nyq = sfreq / 2.0
        cut = min(cut_freq, nyq - 0.5)
        if cut <= 0:
            return data.copy()
        sos = butter(order, cut, btype="low", fs=sfreq, output="sos")
        return sosfiltfilt(sos, data, axis=1).astype(np.float32)
