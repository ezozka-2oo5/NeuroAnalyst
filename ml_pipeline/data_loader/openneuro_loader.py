"""
ml_pipeline/data_loader/openneuro_loader.py
============================================
OpenNeuro platformasidan BIDS formatidagi EEG ma'lumotlarini
avtomatik yuklab olish va lokal diskka tartibli saqlovchi modul.

OpenNeuro DataLad yoki rasmiy CLI orqali kirish imkonini beradi.
pip install datalad openneuro-py
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.config import DataConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)

# OpenNeuro rasmiy S3 bucket URL
OPENNEURO_S3_BASE = "s3://openneuro.org"
# OpenNeuro API base URL
OPENNEURO_API_URL = "https://openneuro.org/crn/datasets"


class OpenNeuroLoader:
    """
    OpenNeuro BIDS EEG datasetlarini yuklab oluvchi klass.

    Ustuvorlik tartibi:
      1. ``openneuro-py`` Python paketi (eng qulay)
      2. ``datalad`` CLI (katta datasetlar uchun optimal)
      3. AWS S3 CLI (``aws s3 sync``) fallback

    Qo'llanilishi::

        loader = OpenNeuroLoader(cfg)
        loader.download("ds002778")
        edf_files = loader.get_local_edf_paths("ds002778")
        metadata = loader.parse_bids_metadata("ds002778")
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.raw_dir = Path(config.raw_data_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def download(
        self,
        dataset_id: str,
        subjects: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        force_redownload: bool = False,
        max_subjects: Optional[int] = None,
    ) -> Path:
        """
        OpenNeuro datasetini BIDS formatida yuklab oladi.

        Args:
            dataset_id: OpenNeuro accession ID (masalan ``"ds002778"``).
            subjects: Yuklanadigan subjectlar ro'yxati (masalan ``["sub-01", "sub-02"]``).
                      None berilsa hammasi yuklanadi.
            modalities: BIDS modalliklar (masalan ``["eeg", "meg"]``).
                        None berilsa EEG bo'limi tanlanadi.
            force_redownload: Mavjud fayllarni qayta yuklash.
            max_subjects: Debug uchun N ta subjectni cheklash.

        Returns:
            Yuklab olingan dataset katalogi.
        """
        if modalities is None:
            modalities = ["eeg"]

        dest_dir = self.raw_dir / "openneuro" / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"OpenNeuro yuklash boshlandi: {dataset_id} → {dest_dir}")

        # Birinchi navbatda openneuro-py sinab ko'ramiz
        try:
            import openneuro as on_pkg
            return self._download_with_openneuro_py(
                on_pkg, dataset_id, dest_dir, subjects, modalities, max_subjects
            )
        except ImportError:
            logger.warning(
                "openneuro-py topilmadi. "
                "datalad urinilmoqda. "
                "Tavsiya: pip install openneuro-py"
            )

        # Keyin DataLad
        if shutil.which("datalad"):
            return self._download_with_datalad(
                dataset_id, dest_dir, subjects, modalities, force_redownload, max_subjects
            )
        else:
            logger.warning("datalad topilmadi. aws s3 urinilmoqda.")

        # Oxirgi fallback — AWS S3
        return self._download_with_aws_s3(
            dataset_id, dest_dir, subjects, modalities, force_redownload, max_subjects
        )

    def get_local_edf_paths(self, dataset_id: str) -> List[Path]:
        """
        Lokal diskdagi BIDS EEG EDF/BDF fayllarini qaytaradi.

        BIDS tuzilmasi::

            dataset_id/
              sub-01/
                eeg/
                  sub-01_task-rest_eeg.edf
                  sub-01_task-rest_eeg.json
              sub-02/
                ...

        Args:
            dataset_id: Dataset accession ID.

        Returns:
            Tartiblangan EDF fayl yo'llari.
        """
        base = self.raw_dir / "openneuro" / dataset_id
        if not base.exists():
            raise FileNotFoundError(
                f"{base} topilmadi. Avval download() ni chaqiring."
            )

        edf_paths = sorted(
            list(base.rglob("*_eeg.edf"))
            + list(base.rglob("*_eeg.bdf"))
            + list(base.rglob("*_eeg.EDF"))
        )
        logger.info(f"{len(edf_paths)} ta BIDS EEG fayl topildi: {base}")
        return edf_paths

    def parse_bids_metadata(self, dataset_id: str) -> Dict[str, dict]:
        """
        BIDS dataset ildizidagi barcha sidecar JSON fayllarini parse qiladi.

        Har bir EDF fayl uchun ``*_eeg.json`` (task, sampling rate, channels)
        va ``*_events.tsv`` (stimulus vaqtlari) fayllarini o'qiydi.

        Args:
            dataset_id: Dataset accession ID.

        Returns:
            ``{edf_fayl_yo'li: {metadata_dict}}`` shaklida lug'at.
        """
        base = self.raw_dir / "openneuro" / dataset_id
        result: Dict[str, dict] = {}

        for edf_path in self.get_local_edf_paths(dataset_id):
            stem = edf_path.stem  # masalan: sub-01_task-rest_eeg
            meta: dict = {}

            # 1) Sidecar JSON (_eeg.json)
            json_path = edf_path.parent / f"{stem}.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as fh:
                    meta.update(json.load(fh))

            # 2) Events TSV (_events.tsv)
            # BIDS naming: sub-01_task-rest_events.tsv
            events_stem = stem.replace("_eeg", "_events")
            events_path = edf_path.parent / f"{events_stem}.tsv"
            if events_path.exists():
                meta["events"] = self._parse_bids_events_tsv(events_path)

            # 3) Channels TSV (_channels.tsv)
            channels_stem = stem.replace("_eeg", "_channels")
            channels_path = edf_path.parent / f"{channels_stem}.tsv"
            if channels_path.exists():
                meta["channels"] = self._parse_bids_channels_tsv(channels_path)

            # 4) Subject-level participants.tsv dan demografik ma'lumot
            participants_path = base / "participants.tsv"
            if participants_path.exists():
                subject_id = self._extract_subject_id(edf_path)
                participant_meta = self._get_participant_info(participants_path, subject_id)
                meta.update(participant_meta)

            result[str(edf_path)] = meta

        return result

    def get_dataset_description(self, dataset_id: str) -> Optional[dict]:
        """
        BIDS ``dataset_description.json`` faylini qaytaradi.

        Bu faylda dataset nomi, versiyasi, mualliflar ro'yxati va
        qo'llaniladigan litsenziya ko'rsatilgan.

        Args:
            dataset_id: Dataset accession ID.

        Returns:
            description dict yoki None (fayl topilmasa).
        """
        desc_path = self.raw_dir / "openneuro" / dataset_id / "dataset_description.json"
        if desc_path.exists():
            with open(desc_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def list_subjects(self, dataset_id: str) -> List[str]:
        """
        Dataset ichidagi barcha subject IDlarini qaytaradi.

        Args:
            dataset_id: Dataset accession ID.

        Returns:
            ``["sub-01", "sub-02", ...]`` kabi ro'yxat.
        """
        base = self.raw_dir / "openneuro" / dataset_id
        subjects = sorted([d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("sub-")])
        return subjects

    # ──────────────────────────────────────────────────────────────
    # Private: download backends
    # ──────────────────────────────────────────────────────────────

    def _download_with_openneuro_py(
        self, on_pkg, dataset_id, dest_dir, subjects, modalities, max_subjects
    ) -> Path:
        """openneuro-py paketi yordamida yuklab olish."""
        logger.info("openneuro-py orqali yuklanmoqda...")
        download_kwargs: dict = {
            "dataset": dataset_id,
            "target_dir": str(dest_dir),
            "include": [],
        }

        # Faqat EEG va metadata fayllarni yuklaymiz
        for modality in modalities:
            download_kwargs["include"].append(f"*/{modality}/*")
        download_kwargs["include"].extend([
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
            "README",
            "CHANGES",
        ])

        if subjects is not None:
            subj_list = subjects[:max_subjects] if max_subjects else subjects
            download_kwargs["include"] = [
                f"{subj}/**" for subj in subj_list
            ] + [
                "dataset_description.json",
                "participants.tsv",
            ]

        try:
            on_pkg.download(**download_kwargs)
            logger.info(f"openneuro-py yuklash muvaffaqiyatli tugadi: {dest_dir}")
        except Exception as exc:
            logger.error(f"openneuro-py xatosi: {exc}")
            raise

        return dest_dir

    def _download_with_datalad(
        self, dataset_id, dest_dir, subjects, modalities, force, max_subjects
    ) -> Path:
        """DataLad CLI orqali yuklash."""
        logger.info("datalad orqali yuklanmoqda...")
        openneuro_url = f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"

        # 1) DataLad install (dataset klonlash)
        if not (dest_dir / ".datalad").exists():
            logger.info(f"datalad install: {openneuro_url}")
            self._run_command(
                ["datalad", "install", "--source", openneuro_url, str(dest_dir)],
                timeout=300,
            )

        # 2) Kerakli fayllarni get qilish
        target_paths = []
        if subjects is not None:
            subj_list = subjects[:max_subjects] if max_subjects else subjects
            for subj in subj_list:
                for modality in modalities:
                    target_paths.append(str(dest_dir / subj / modality))
        else:
            # Hammasi
            for modality in modalities:
                target_paths.append(str(dest_dir / "*" / modality))

        if not target_paths:
            target_paths = [str(dest_dir)]

        for path in target_paths:
            logger.info(f"datalad get: {path}")
            self._run_command(
                ["datalad", "get", "--jobs=4", path],
                timeout=7200,  # 2 soat
            )

        return dest_dir

    def _download_with_aws_s3(
        self, dataset_id, dest_dir, subjects, modalities, force, max_subjects
    ) -> Path:
        """AWS S3 CLI yordamida to'g'ridan-to'g'ri S3 bucketdan yuklash."""
        logger.info("AWS S3 CLI orqali yuklanmoqda (anonim kirish)...")
        s3_path = f"{OPENNEURO_S3_BASE}/{dataset_id}/"

        cmd = [
            "aws", "s3", "sync",
            s3_path, str(dest_dir),
            "--no-sign-request",       # Anonim kirish (public dataset)
            "--exclude", "*",
            "--include", "*.edf",
            "--include", "*.bdf",
            "--include", "*.json",
            "--include", "*.tsv",
            "--include", "dataset_description.json",
            "--include", "participants.tsv",
        ]

        if subjects is not None:
            # S3 filter: faqat ma'lum subjectlar
            for subj in (subjects[:max_subjects] if max_subjects else subjects):
                cmd.extend(["--include", f"{subj}/*"])

        self._run_command(cmd, timeout=7200)
        return dest_dir

    def _run_command(self, cmd: List[str], timeout: int = 300) -> None:
        """Shell buyrug'ini ishga tushiradi va xatolarni log qiladi."""
        logger.debug(f"Buyruq: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.warning(
                f"Buyruq xato: exit={result.returncode}\n"
                f"  stdout: {result.stdout[:300]}\n"
                f"  stderr: {result.stderr[:300]}"
            )
        else:
            logger.debug("Buyruq muvaffaqiyatli.")

    # ──────────────────────────────────────────────────────────────
    # Private: BIDS parsing helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_bids_events_tsv(tsv_path: Path) -> List[dict]:
        """
        BIDS events.tsv ni parse qilib, har bir event uchun lug'at qaytaradi.

        TSV sarlavha qatori: onset  duration  trial_type  ...
        """
        events: List[dict] = []
        with open(tsv_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        if not lines:
            return events

        headers = lines[0].strip().split("\t")
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) == len(headers):
                events.append(dict(zip(headers, parts)))

        return events

    @staticmethod
    def _parse_bids_channels_tsv(tsv_path: Path) -> List[dict]:
        """
        BIDS channels.tsv ni parse qiladi.

        TSV sarlavha qatori: name  type  units  sampling_frequency  ...
        """
        channels: List[dict] = []
        with open(tsv_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        if not lines:
            return channels

        headers = lines[0].strip().split("\t")
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if parts:
                channels.append(dict(zip(headers, parts)))

        return channels

    @staticmethod
    def _extract_subject_id(edf_path: Path) -> str:
        """
        BIDS fayl nomidan subject ID ni ajratib oladi.
        Masalan: ``sub-01_task-rest_eeg.edf`` → ``"sub-01"``
        """
        name = edf_path.stem  # sub-01_task-rest_eeg
        parts = name.split("_")
        for part in parts:
            if part.startswith("sub-"):
                return part
        # Fallback: parent papka nomi
        parent = edf_path.parent.parent
        if parent.name.startswith("sub-"):
            return parent.name
        return "sub-unknown"

    @staticmethod
    def _get_participant_info(participants_tsv: Path, subject_id: str) -> dict:
        """
        participants.tsv dan bitta subjectning demografik ma'lumotini oladi.
        (Yosh, jins, guruh va h.k.)
        """
        try:
            with open(participants_tsv, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            if not lines:
                return {}
            headers = lines[0].strip().split("\t")
            for line in lines[1:]:
                parts = line.strip().split("\t")
                if parts and parts[0] == subject_id:
                    return dict(zip(headers, parts))
        except (OSError, ValueError):
            pass
        return {}


# ──────────────────────────────────────────────────────────────────
# shutil import (module top-levelga ko'tarmadik — conditional)
# ──────────────────────────────────────────────────────────────────
import shutil  # noqa: E402  (pastda ishlatiladi)
