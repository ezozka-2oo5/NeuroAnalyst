"""
ml_pipeline/data_loader/bids_parser.py
========================================
BIDS (Brain Imaging Data Structure) formatidagi EEG datasetlarini
MNE kutubxonasi orqali yuklab, standart formatga keltiruvchi modul.

BIDS - bu neyroimaging ma'lumotlari uchun rasmiy standart:
https://bids-specification.readthedocs.io/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.config import DataConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BIDSParser:
    """
    BIDS formatidagi EEG datasetini MNE orqali yuklaydi va
    pipeline uchun standart numpy array formatiga keltiradi.

    Natija strukturasi::

        {
            "data": np.ndarray,          # shape: (n_channels, n_samples) µV
            "sfreq": float,              # Hz
            "channel_names": List[str],  # ["Fp1", "Fp2", ...]
            "subject_id": str,
            "task": str,
            "label": int,
            "metadata": dict,
        }
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self._check_mne()

    def _check_mne(self) -> None:
        """MNE mavjudligini tekshiradi."""
        try:
            import mne  # noqa: F401
        except ImportError:
            raise ImportError(
                "MNE-Python o'rnatilmagan. "
                "EDF/BDF fayllarini o'qish uchun zarur.\n"
                "O'rnatish: pip install mne"
            )

    def load_edf(
        self,
        edf_path: Path,
        metadata: Optional[dict] = None,
        preload: bool = True,
    ) -> Optional[dict]:
        """
        Bitta EDF/BDF faylini MNE orqali yuklaydi.

        Args:
            edf_path: EDF fayl yo'li.
            metadata: BIDS sidecar JSON dan olingan metadata (ixtiyoriy).
            preload: True bo'lsa, butun signal RAM ga yuklanadi.

        Returns:
            Standart recording dict yoki None (xato bo'lsa).
        """
        import mne

        edf_path = Path(edf_path)
        if not edf_path.exists():
            logger.error(f"EDF fayl topilmadi: {edf_path}")
            return None

        suffix = edf_path.suffix.lower()
        if suffix == ".bdf":
            reader = mne.io.read_raw_bdf
        elif suffix == ".set":
            reader = mne.io.read_raw_eeglab
        else:
            reader = mne.io.read_raw_edf

        try:
            raw = reader(str(edf_path), preload=preload, verbose="ERROR")
        except Exception as exc:
            logger.warning(f"EDF o'qilmadi ({edf_path.name}): {exc}")
            return None

        # EEG kanallarini tanlash
        raw = self._select_eeg_channels(raw)
        if raw is None or len(raw.ch_names) == 0:
            logger.warning(f"EEG kanallar topilmadi: {edf_path.name}")
            return None

        # Kanallarni tozalash (tartibga solish)
        raw = self._standardize_channel_names(raw)

        # Ma'lumotlarni µV ga aylantirish
        data = raw.get_data() * 1e6  # Volt → µV

        recording = {
            "data": data.astype(np.float32),
            "sfreq": float(raw.info["sfreq"]),
            "channel_names": list(raw.ch_names),
            "source_path": str(edf_path),
            "source_name": edf_path.name,
            "source_type": "bids_edf" if metadata else "edf",
            "subject_id": self._extract_bids_entity(edf_path.stem, "sub"),
            "task": self._extract_bids_entity(edf_path.stem, "task"),
            "session": self._extract_bids_entity(edf_path.stem, "ses"),
            "run": self._extract_bids_entity(edf_path.stem, "run"),
            "label": -1,          # Label keyinroq to'ldiriladi
            "metadata": metadata or {},
        }

        return recording

    def load_dataset(
        self,
        dataset_dir: Path,
        label_map: Optional[Dict[str, int]] = None,
        subjects: Optional[List[str]] = None,
        max_files: Optional[int] = None,
    ) -> List[dict]:
        """
        Butun BIDS dataset katalogini yuklaydi.

        Args:
            dataset_dir: BIDS dataset ildiz katalogi.
            label_map: Subjectning diagnostik guruhi → label int mapping.
                       Bu ko'pincha participants.tsv dagi "group" kolonkasidan olinadi.
            subjects: Faqat ma'lum subjectlarni yuklash.
            max_files: Debug uchun N ta fayl bilan cheklash.

        Returns:
            Recording dict ro'yxati.
        """
        dataset_dir = Path(dataset_dir)

        # participants.tsv dan label mappingni o'qiymiz
        participant_labels = self._load_participant_labels(dataset_dir, label_map)

        # Barcha EEG fayllarni topamiz (EDF, BDF, SET)
        all_edf = sorted(
            list(dataset_dir.rglob("*_eeg.edf"))
            + list(dataset_dir.rglob("*_eeg.bdf"))
            + list(dataset_dir.rglob("*_eeg.set"))
        )

        if subjects is not None:
            all_edf = [
                p for p in all_edf
                if self._extract_bids_entity(p.stem, "sub") in subjects
            ]

        if max_files is not None:
            all_edf = all_edf[:max_files]

        logger.info(f"Jami {len(all_edf)} ta EDF fayl yuklanadi: {dataset_dir}")

        recordings: List[dict] = []
        for idx, edf_path in enumerate(all_edf):
            logger.info(f"  [{idx + 1}/{len(all_edf)}] {edf_path.name}")

            # Sidecar JSON metadata
            meta = self._load_sidecar_json(edf_path)

            recording = self.load_edf(edf_path, metadata=meta)
            if recording is None:
                continue

            # Label o'rnatish
            subj_id = recording["subject_id"]
            recording["label"] = participant_labels.get(subj_id, -1)

            recordings.append(recording)

        labeled = sum(1 for r in recordings if r["label"] != -1)
        logger.info(
            f"Yuklandi: {len(recordings)} ta recording, "
            f"{labeled} ta labellangan."
        )
        return recordings

    def extract_events_from_raw(
        self,
        edf_path: Path,
        event_to_label: Dict[str, int],
    ) -> List[Tuple[int, int]]:
        """
        EDF faylidagi MNE annotation lardan event vaqtlari va labellarni oladi.

        Args:
            edf_path: EDF fayl yo'li.
            event_to_label: Annotation string → label int mapping.

        Returns:
            ``[(sample_index, label), ...]`` ro'yxati.
        """
        import mne

        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        except Exception as exc:
            logger.warning(f"Annotation o'qilmadi ({edf_path.name}): {exc}")
            return []

        events_list: List[Tuple[int, int]] = []
        sfreq = raw.info["sfreq"]

        for annotation in raw.annotations:
            desc = annotation["description"].strip()
            onset_sec = annotation["onset"]
            duration_sec = annotation["duration"]

            label = event_to_label.get(desc)
            if label is None:
                # Partial match: masalan "seiz" "fnsz" ichida ham bo'lishi mumkin
                for key, val in event_to_label.items():
                    if key in desc.lower() or desc.lower() in key:
                        label = val
                        break

            if label is not None:
                onset_sample = int(onset_sec * sfreq)
                duration_samples = int(duration_sec * sfreq)
                events_list.append((onset_sample, label, duration_samples))

        logger.debug(f"{len(events_list)} ta event topildi: {edf_path.name}")
        return events_list

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _select_eeg_channels(self, raw):
        """Faqat EEG kanallarini qoldiradi, EMG/EOG/EKG ni olib tashlaydi."""
        import mne

        try:
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            if len(eeg_picks) == 0:
                # EEG tipi belgilanmagan bo'lishi mumkin
                # Barcha kanallarni olishga harakat qilamiz
                logger.debug("EEG tipi topilmadi, hamma kanallar ishlatiladi.")
                return raw

            raw.pick(eeg_picks)
        except Exception as exc:
            logger.debug(f"Kanal tanlashda xato (o'tkazildi): {exc}")

        return raw

    def _standardize_channel_names(self, raw):
        """
        Kanal nomlarini standart 10-20 formatiga keltiradi.

        Turli qurilmalar turlicha nomlaydi:
        ``EEG Fp1-REF`` → ``Fp1``
        ``EEG FP1`` → ``Fp1``
        """
        rename_map = {}
        for ch in raw.ch_names:
            clean = ch.strip()
            # "EEG " prefiksini olib tashlaymiz
            for prefix in ("EEG ", "eeg ", "EEG-", "EEG_"):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
            # "-REF", "-LE" suffixlarini olib tashlaymiz
            for suffix in ("-REF", "-LE", "-AVG", "-A1", "-A2", "-M1", "-M2"):
                if clean.endswith(suffix):
                    clean = clean[: -len(suffix)]
            # Birinchi harf katta, qolganlari kichik (Fp1, Cz, POz)
            clean = clean.strip()
            if clean and clean[0].isalpha():
                clean = clean[0].upper() + clean[1:].lower()

            if clean != ch:
                rename_map[ch] = clean

        if rename_map:
            raw.rename_channels(rename_map)

        return raw

    @staticmethod
    def _extract_bids_entity(stem: str, entity: str) -> str:
        """
        BIDS fayl nomidan entity ni ajratib oladi.

        ``sub-01_task-rest_eeg`` dan ``sub`` entity = ``"01"``.

        Args:
            stem: Fayl nomi (kengaytikmassiz).
            entity: BIDS entity: ``"sub"``, ``"task"``, ``"ses"``, ``"run"``.

        Returns:
            Entity qiymati yoki bo'sh string.
        """
        for part in stem.split("_"):
            if part.startswith(f"{entity}-"):
                return part[len(entity) + 1:]
        return ""

    @staticmethod
    def _load_sidecar_json(edf_path: Path) -> dict:
        """EDF fayl bilan bir xil nomli _eeg.json faylini yuklaydi."""
        json_path = edf_path.with_suffix(".json")
        # Masalan: sub-01_task-rest_eeg.edf → sub-01_task-rest_eeg.json
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    @staticmethod
    def _load_participant_labels(
        dataset_dir: Path,
        label_map: Optional[Dict[str, int]],
    ) -> Dict[str, int]:
        """
        participants.tsv dan har bir subjectning labelini o'qiydi.

        Agar label_map ``{"A": 0, "F": 1, "C": 2}`` berilgan bo'lsa,
        ``participants.tsv`` dagi ``group`` kolonkasidagi qiymatlarni
        label integerga aylantiradi.

        Returns:
            ``{"sub-01": 0, "sub-02": 1, ...}``
        """
        participants_tsv = dataset_dir / "participants.tsv"
        result: Dict[str, int] = {}

        if not participants_tsv.exists() or label_map is None:
            return result

        try:
            with open(participants_tsv, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return result

        if len(lines) < 2:
            return result

        headers = lines[0].strip().split("\t")
        id_col = 0
        group_col = None

        # "group" yoki "diagnosis" kolonkasini topamiz
        for possible in ("group", "Group", "diagnosis", "Diagnosis", "pathology"):
            if possible in headers:
                group_col = headers.index(possible)
                break

        if group_col is None:
            logger.warning(
                "participants.tsv da 'group'/'diagnosis' kolonkasi topilmadi. "
                "Labellar o'rnatilmaydi."
            )
            return result

        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) > max(id_col, group_col):
                subject_id = parts[id_col].replace("sub-", "")
                group_str = parts[group_col]
                label = label_map.get(group_str)
                if label is not None:
                    result[subject_id] = label

        return result
