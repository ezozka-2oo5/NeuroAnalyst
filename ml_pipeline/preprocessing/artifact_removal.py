"""
ml_pipeline/preprocessing/artifact_removal.py
================================================
EEG signallaridagi artefaktlarni (ko'z qisish, yurak urishi, muskullar)
avtomatik aniqlash va olib tashlash moduli.

ICA (Independent Component Analysis) asosida amalga oshiriladi.
MNE-Python kutubxonasidan foydalanadi.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.config import PreprocessingConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ArtifactRemover:
    """
    EEG signallaridagi artefaktlarni aniqlash va olib tashlash.

    Usullar:
    1. **ICA** — EOG (ko'z), ECG (yurak) artefaktlarini ajratib olib tashlaydi.
    2. **Amplitude threshold** — Haddan tashqari katta amplitudali segmentlarni belgilaydi.
    3. **Flat channel detection** — Signali yo'q (o'lik) kanallarni aniqlaydi.
    4. **Noisy channel detection** — Haddan tashqari shovqinli kanallarni aniqlaydi.

    Qo'llanilishi::

        cfg = PreprocessingConfig()
        remover = ArtifactRemover(cfg)
        clean_raw = remover.apply_ica(mne_raw)
        bad_channels = remover.detect_bad_channels(data, sfreq)
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    # ──────────────────────────────────────────────────────────────
    # ICA-based artifact removal (MNE orqali)
    # ──────────────────────────────────────────────────────────────

    def apply_ica(
        self,
        raw,   # mne.io.BaseRaw
        eog_channel: Optional[str] = None,
        ecg_channel: Optional[str] = None,
        n_components: Optional[int] = None,
    ):
        """
        MNE Raw obyektiga ICA qo'llab, EOG/ECG artefaktlarini olib tashlaydi.

        ICA algoritmi:
          1. Signalni mustaqil komponentlarga ajratadi.
          2. EOG/ECG bilan korrelyatsiyasi yuqori komponentlarni belgilaydi.
          3. Belgilangan komponentlarni o'chirib, signalni tiklaydi.

        Args:
            raw: mne.io.RawEDF yoki shunga o'xshash MNE Raw obyekti.
            eog_channel: EOG kanal nomi (masalan "EOG" yoki "Fp1").
                         None bo'lsa MNE avtomatik topishga harakat qiladi.
            ecg_channel: ECG kanal nomi. None bo'lsa siqilmaydi.
            n_components: ICA komponent soni.
                          None bo'lsa config dan olinadi.

        Returns:
            ICA qo'llangan MNE Raw obyekti (o'zgartirish o'sha obyektga qo'llanadi).
        """
        if not self.config.apply_ica:
            return raw

        try:
            import mne
        except ImportError:
            logger.warning("MNE topilmadi. ICA o'tkaziladi.")
            return raw

        n_comp = n_components or self.config.ica_n_components
        method = self.config.ica_method
        max_iter = self.config.ica_max_iter

        # Kanallar sonini tekshiramiz
        n_eeg = len(mne.pick_types(raw.info, eeg=True))
        if n_eeg < n_comp:
            n_comp = n_eeg - 1
            logger.info(f"ICA n_components kamaytildi: {n_comp} (kanal soni: {n_eeg})")

        if n_comp < 2:
            logger.warning("Kanal soni juda kam, ICA qo'llanilmaydi.")
            return raw

        logger.info(
            f"ICA boshlandi: method={method}, n_components={n_comp}, max_iter={max_iter}"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica = mne.preprocessing.ICA(
                n_components=n_comp,
                method=method,
                max_iter=max_iter,
                random_state=42,
            )
            # Bandpass filtrlangan data yaxshiroq ishlaydi
            raw_filtered = raw.copy().filter(1.0, 40.0, verbose="ERROR")
            ica.fit(raw_filtered, verbose="ERROR")

        excluded_components: List[int] = []

        # EOG artefaktlari
        eog_chs = self._find_eog_channels(raw, eog_channel)
        if eog_chs:
            try:
                eog_idx, eog_scores = ica.find_bads_eog(
                    raw, ch_name=eog_chs[0], threshold=3.0, verbose="ERROR"
                )
                excluded_components.extend(eog_idx)
                logger.info(f"EOG artefakt komponentlari: {eog_idx}")
            except Exception as exc:
                logger.debug(f"EOG topishda xato: {exc}")

        # ECG artefaktlari
        ecg_chs = self._find_ecg_channels(raw, ecg_channel)
        if ecg_chs:
            try:
                ecg_idx, ecg_scores = ica.find_bads_ecg(
                    raw, ch_name=ecg_chs[0], threshold=3.0, verbose="ERROR"
                )
                excluded_components.extend(ecg_idx)
                logger.info(f"ECG artefakt komponentlari: {ecg_idx}")
            except Exception as exc:
                logger.debug(f"ECG topishda xato: {exc}")

        # Avtokor EOG (kanal bo'lmasa ham) — Fp1/Fp2 yordamida
        if not eog_chs and not ecg_chs:
            frontal_chs = [ch for ch in raw.ch_names if ch.lower() in ("fp1", "fp2")]
            if frontal_chs:
                try:
                    eog_idx, _ = ica.find_bads_eog(
                        raw, ch_name=frontal_chs[0], threshold=3.5, verbose="ERROR"
                    )
                    excluded_components.extend(eog_idx)
                    logger.info(f"Fp1/Fp2 yordamida EOG: {eog_idx}")
                except Exception:
                    pass

        # Noyob komponentlar
        ica.exclude = list(set(excluded_components))

        if ica.exclude:
            logger.info(f"O'chirilayotgan komponentlar: {ica.exclude}")
            ica.apply(raw, verbose="ERROR")
        else:
            logger.info("Artefakt komponent topilmadi. ICA o'zgartirish qilmadi.")

        return raw

    # ──────────────────────────────────────────────────────────────
    # Amplitude threshold
    # ──────────────────────────────────────────────────────────────

    def detect_amplitude_artifacts(
        self,
        data: np.ndarray,
        threshold_uv: float = 150.0,
    ) -> np.ndarray:
        """
        Har bir namuna uchun amplitude artefakt maskasini qaytaradi.

        Agar birorta kanalda qiymat threshold dan oshsa, o'sha namuna artefakt.

        Args:
            data: (n_channels, n_samples) µV.
            threshold_uv: Maksimal ruxsat etilgan amplituda µV.

        Returns:
            Boolean mask, shape (n_samples,): True = artefakt.
        """
        # Har bir kanalda threshold dan oshgan namunalar
        above_threshold = np.any(np.abs(data) > threshold_uv, axis=0)
        ratio = np.mean(above_threshold) * 100
        logger.debug(f"Amplitude artefakt: {ratio:.1f}% namuna ({threshold_uv} µV threshold)")
        return above_threshold

    def compute_artifact_ratio(
        self,
        data: np.ndarray,
        sfreq: float,
    ) -> Dict[str, float]:
        """
        Signal sifati ko'rsatkichlarini hisoblaydi.

        Args:
            data: (n_channels, n_samples) µV.
            sfreq: Sampling chastotasi Hz.

        Returns:
            Dict: {
                "amplitude_ratio": 0-100 foiz,
                "flat_channel_ratio": 0-1,
                "signal_quality_score": 0-100,
            }
        """
        from scipy.signal import welch

        n_channels, n_samples = data.shape

        # 1. Amplitude threshold
        amp_mask = self.detect_amplitude_artifacts(
            data, self.config.epoch_reject_threshold_uv
        )
        amplitude_ratio = float(np.mean(amp_mask)) * 100

        # 2. Flat channels (std < threshold)
        ch_std = np.std(data, axis=1)
        flat_mask = ch_std < self.config.flat_channel_threshold_uv
        flat_ratio = float(np.mean(flat_mask))

        # 3. Shovqin baholash: Robust Z-score
        centered = data - np.median(data, axis=1, keepdims=True)
        mad = np.median(np.abs(centered), axis=1, keepdims=True) + 1e-9
        robust_z = 0.6745 * centered / mad
        noisy_ratio = float(np.mean(np.abs(robust_z) > 6.0)) * 100

        # 4. Umumiy sifat balli
        quality = float(
            np.clip(
                100.0
                - amplitude_ratio * 0.5
                - noisy_ratio * 0.3
                - flat_ratio * 30.0,
                5.0,
                100.0,
            )
        )

        return {
            "amplitude_ratio": amplitude_ratio,
            "flat_channel_ratio": flat_ratio,
            "noisy_ratio": noisy_ratio,
            "signal_quality_score": quality,
        }

    # ──────────────────────────────────────────────────────────────
    # Bad channel detection (numpy, MNE faqat ixtiyoriy)
    # ──────────────────────────────────────────────────────────────

    def detect_bad_channels(
        self,
        data: np.ndarray,
        sfreq: float,
        channel_names: Optional[List[str]] = None,
    ) -> Tuple[List[int], List[str]]:
        """
        Flat va shovqinli kanallarni aniqlaydi.

        Args:
            data: (n_channels, n_samples) µV.
            sfreq: Hz.
            channel_names: Kanal nomlari ro'yxati (ixtiyoriy, logging uchun).

        Returns:
            (bad_indices, bad_names) tuple.
        """
        n_channels = data.shape[0]
        bad_indices: List[int] = []

        ch_std = np.std(data, axis=1)
        ch_median_abs_dev = np.median(np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1)

        # Global MAD (barcha kanallar o'rtacha)
        global_mad = np.median(ch_median_abs_dev) + 1e-9

        for idx in range(n_channels):
            name = channel_names[idx] if channel_names else str(idx)

            # Flat kanal
            if ch_std[idx] < self.config.flat_channel_threshold_uv:
                logger.debug(f"Flat kanal: {name} (std={ch_std[idx]:.2f} µV)")
                bad_indices.append(idx)
                continue

            # Shovqinli kanal (MAD z-score)
            z = ch_median_abs_dev[idx] / global_mad
            if z > self.config.noisy_channel_threshold_z:
                logger.debug(f"Shovqinli kanal: {name} (z={z:.2f})")
                bad_indices.append(idx)

        bad_names = (
            [channel_names[i] for i in bad_indices]
            if channel_names
            else [str(i) for i in bad_indices]
        )
        if bad_indices:
            logger.info(f"{len(bad_indices)} ta yomon kanal aniqlandi: {bad_names}")

        return bad_indices, bad_names

    def interpolate_bad_channels(
        self,
        data: np.ndarray,
        bad_indices: List[int],
        channel_names: List[str],
    ) -> np.ndarray:
        """
        Yomon kanallarni qo'shni kanallar o'rtachasidan interpolatsiya qiladi.

        Bu oddiy nearest-neighbor interpolatsiya.
        MNE orqali to'liq sferik interpolatsiya uchun apply_ica dan foydalaning.

        Args:
            data: (n_channels, n_samples) µV.
            bad_indices: Yomon kanal indekslari.
            channel_names: Kanal nomlari (10-20 standart tartibi muhim).

        Returns:
            Interpolatsiya qilingan data.
        """
        if not bad_indices:
            return data

        clean_data = data.copy()
        good_indices = [i for i in range(data.shape[0]) if i not in bad_indices]

        for bad_idx in bad_indices:
            # Eng yaqin 2 ta yaxshi kanaldan o'rtacha
            distances = [abs(bad_idx - g) for g in good_indices]
            nearest = sorted(zip(distances, good_indices))[:2]
            if nearest:
                weights = [1.0 / (d + 1) for d, _ in nearest]
                total_w = sum(weights)
                interp = sum(
                    w * data[idx] for w, (_, idx) in zip(weights, nearest)
                ) / total_w
                clean_data[bad_idx] = interp
                logger.debug(
                    f"Kanal {channel_names[bad_idx]} interpolatsiya qilindi "
                    f"(manbalar: {[channel_names[_] for _, (_d, _) in zip(weights, nearest)]})"
                )

        return clean_data

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _find_eog_channels(raw, preferred: Optional[str]) -> List[str]:
        """MNE Raw obyektidan EOG kanallarini topadi."""
        if preferred and preferred in raw.ch_names:
            return [preferred]
        # Odatiy EOG kanal nomlari
        for candidate in ("EOG", "EOG Horizontal", "EOG Vertical", "HEOG", "VEOG"):
            if candidate in raw.ch_names:
                return [candidate]
        return []

    @staticmethod
    def _find_ecg_channels(raw, preferred: Optional[str]) -> List[str]:
        """MNE Raw obyektidan ECG kanallarini topadi."""
        if preferred and preferred in raw.ch_names:
            return [preferred]
        for candidate in ("ECG", "EKG"):
            if candidate in raw.ch_names:
                return [candidate]
        return []
