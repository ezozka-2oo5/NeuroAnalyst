"""
ml_pipeline/data_loader/physionet_loader.py
============================================
PhysioNet platformasidan EEG ma'lumotlarini avtomatik yuklab olish
va lokal diskka tartibli saqlovchi modul.

PhysioNet wfdb (WaveForm DataBase) kutubxonasidan foydalanadi.
pip install wfdb
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.config import DataConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PhysioNetLoader:
    """
    PhysioNet EEG datasetlarini yuklab olish va EDF/annotation fayllarini
    lokal disk strukturasiga tartibli joylashtiruvchi klass.

    Qo'llanilishi::

        cfg = DataConfig()
        loader = PhysioNetLoader(cfg)
        loader.download("chb-mit-scalp-eeg-database-1.0.0")
        files = loader.get_local_edf_paths("chb-mit-scalp-eeg-database-1.0.0")
    """

    # PhysioNet public S3-mirror URL (rasmiy API endpoint)
    BASE_URL = "https://physionet.org/files"

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.raw_dir = Path(config.raw_data_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def download(
        self,
        dataset_slug: str,
        subjects: Optional[List[str]] = None,
        force_redownload: bool = False,
        max_files: Optional[int] = None,
    ) -> Path:
        """
        PhysioNet datasetini yuklab oladi.

        Agar wfdb o'rnatilmagan bo'lsa wget/curl orqali fallback qiladi.

        Args:
            dataset_slug: PhysioNet content slug (masalan
                          ``"chb-mit-scalp-eeg-database-1.0.0"``).
            subjects: Faqat ma'lum subjectlarni yuklash uchun ID ro'yxati.
                      None berilsa, hammasini yuklab oladi.
            force_redownload: True bo'lsa, mavjud fayllarni qayta yuklaydi.
            max_files: Debug uchun faqat N ta fayl yuklash.

        Returns:
            Yuklab olingan ma'lumotlar katalogi.
        """
        dest_dir = self.raw_dir / "physionet" / dataset_slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PhysioNet yuklash boshlandi: {dataset_slug} → {dest_dir}")

        try:
            import wfdb
            return self._download_with_wfdb(
                wfdb, dataset_slug, dest_dir, subjects, force_redownload, max_files
            )
        except ImportError:
            logger.warning(
                "wfdb kutubxonasi topilmadi. "
                "Wget yordamida yuklab olish uriniladi. "
                "Tavsiya: pip install wfdb"
            )
            return self._download_with_wget(
                dataset_slug, dest_dir, subjects, force_redownload, max_files
            )

    def get_local_edf_paths(self, dataset_slug: str) -> List[Path]:
        """
        Lokal diskdagi barcha EDF/BDF fayllarini qaytaradi.

        Args:
            dataset_slug: Dataset nomi.

        Returns:
            EDF fayl yo'llari ro'yxati (tartiblangan).
        """
        base = self.raw_dir / "physionet" / dataset_slug
        if not base.exists():
            raise FileNotFoundError(
                f"{base} topilmadi. Avval download() ni chaqiring."
            )
        paths = sorted(
            list(base.rglob("*.edf")) + list(base.rglob("*.EDF"))
            + list(base.rglob("*.bdf")) + list(base.rglob("*.BDF"))
        )
        logger.info(f"{len(paths)} ta EDF/BDF fayl topildi: {base}")
        return paths

    def get_annotation_file(self, edf_path: Path) -> Optional[Path]:
        """
        EDF fayliga mos annotation faylini topadi.

        PhysioNet odatda bir xil nom bilan .seizures yoki .txt yoki
        _annotations.tsv fayllarini saqlaydi.

        Args:
            edf_path: EDF fayl yo'li.

        Returns:
            Annotation fayl yo'li yoki None.
        """
        stem = edf_path.stem
        parent = edf_path.parent

        # CHB-MIT: chb01_01.seizures
        seizure_file = parent / f"{stem}.seizures"
        if seizure_file.exists():
            return seizure_file

        # Umumiy: summary.txt yoki dataset-level annotation
        for ext in (".txt", "_annotations.txt"):
            candidate = parent / f"{stem}{ext}"
            if candidate.exists():
                return candidate

        # Dataset ildizidagi summary fayl
        for summary_name in ("RECORDS-WITH-SEIZURES", "summary.txt"):
            candidate = parent.parent / summary_name
            if candidate.exists():
                return candidate

        return None

    def parse_chbmit_seizure_annotations(
        self, dataset_slug: str
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        CHB-MIT formatidagi seizure annotatsiyalarini parse qiladi.

        Har bir EDF fayli uchun seizure boshlanish va tugash vaqtlarini (soniyada)
        qaytaradi.

        Args:
            dataset_slug: Dataset nomi (odatda "chb-mit-scalp-eeg-database-1.0.0").

        Returns:
            ``{edf_fayl_nomi: [(start_sec, end_sec), ...]}`` shaklida lug'at.
        """
        base = self.raw_dir / "physionet" / dataset_slug
        seizure_map: Dict[str, List[Tuple[float, float]]] = {}

        # Har bir patient papkasidagi summary faylni o'qiymiz
        for summary_path in sorted(base.rglob("*summary*")):
            self._parse_single_chbmit_summary(summary_path, seizure_map)

        logger.info(
            f"CHB-MIT annotation parse: {len(seizure_map)} fayl uchun seizure vaqtlari topildi."
        )
        return seizure_map

    def parse_eegmmidb_events(
        self, dataset_slug: str
    ) -> Dict[str, Dict[str, int]]:
        """
        EEG Motor Movement/Imagery datasetidagi event teglarini qaytaradi.

        PhysioNet EEGMMIDB da har bir yozuv annotation EDF faylida saqlangan.
        Teglar: T0 (rest), T1 (chap qo'l), T2 (o'ng qo'l) — raqamga aylantiriladi.

        Args:
            dataset_slug: Dataset nomi.

        Returns:
            ``{edf_fayl_nomi: {"event_label": label_int}}``
        """
        # EEGMMIDB fayllarining nomi: S{subject_id}R{run_id}.edf
        # Runlar 3,7,11 → left/right fist; 4,8,12 → imagery; 5,9,13 → both fist
        # Bu konventsiya dataset readme dan olingan
        run_label_map = {
            "01": {"T0": 0},                    # baseline (eyes open)
            "02": {"T0": 0},                    # baseline (eyes closed)
            "03": {"T0": 0, "T1": 1, "T2": 2}, # L/R fist actual movement
            "04": {"T0": 0, "T1": 1, "T2": 2}, # L/R fist imagined
            "05": {"T0": 0, "T1": 3, "T2": 4}, # both fist/feet actual
            "06": {"T0": 0, "T1": 3, "T2": 4}, # both fist/feet imagined
            "07": {"T0": 0, "T1": 1, "T2": 2},
            "08": {"T0": 0, "T1": 1, "T2": 2},
            "09": {"T0": 0, "T1": 3, "T2": 4},
            "10": {"T0": 0, "T1": 3, "T2": 4},
            "11": {"T0": 0, "T1": 1, "T2": 2},
            "12": {"T0": 0, "T1": 1, "T2": 2},
            "13": {"T0": 0, "T1": 3, "T2": 4},
            "14": {"T0": 0, "T1": 3, "T2": 4},
        }
        base = self.raw_dir / "physionet" / dataset_slug
        result: Dict[str, Dict[str, int]] = {}

        for edf_path in base.rglob("*.edf"):
            stem = edf_path.stem.upper()  # masalan S001R03
            # Run raqamini ajratib olamiz
            if len(stem) >= 7:
                run_id = stem[-2:]
                label_map = run_label_map.get(run_id, {"T0": 0, "T1": 1, "T2": 2})
                result[str(edf_path)] = label_map

        return result

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _download_with_wfdb(
        self,
        wfdb,
        dataset_slug: str,
        dest_dir: Path,
        subjects: Optional[List[str]],
        force: bool,
        max_files: Optional[int],
    ) -> Path:
        """wfdb kutubxonasi yordamida yuklash."""
        try:
            record_list = wfdb.get_record_list(dataset_slug)
        except Exception as exc:
            logger.error(f"Record ro'yxatini olishda xato: {exc}")
            raise

        if subjects is not None:
            record_list = [
                r for r in record_list
                if any(subj in r for subj in subjects)
            ]
            logger.info(f"Subject filteri qo'llandi: {len(record_list)} ta yozuv qoldi.")

        if max_files is not None:
            record_list = record_list[:max_files]
            logger.info(f"Debug cheklovi: faqat {max_files} ta fayl yuklanadi.")

        total = len(record_list)
        logger.info(f"Jami {total} ta yozuv yuklanadi.")

        for idx, record in enumerate(record_list, start=1):
            record_path = str(dest_dir / record)
            # Allaqachon yuklab olingan bo'lsa o'tkazib yuboramiz
            if not force and self._record_exists(dest_dir, record):
                logger.debug(f"[{idx}/{total}] O'tkazildi (mavjud): {record}")
                continue

            try:
                logger.info(f"[{idx}/{total}] Yuklanmoqda: {record}")
                os.makedirs(os.path.dirname(record_path), exist_ok=True)
                wfdb.dl_files(dataset_slug, str(dest_dir), [record])
                # Serverni haddan ko'p yuklamamaslik uchun kichik tanaffus
                time.sleep(0.5)
            except Exception as exc:
                logger.warning(f"  Xato ({record}): {exc}. O'tkazildi.")

        return dest_dir

    def _download_with_wget(
        self,
        dataset_slug: str,
        dest_dir: Path,
        subjects: Optional[List[str]],
        force: bool,
        max_files: Optional[int],
    ) -> Path:
        """
        wfdb mavjud bo'lmaganda wget bilan yuklash (fallback).

        Bu usul faqat kichik datasetlar uchun ishlatish tavsiya etiladi.
        Katta datasetlar uchun avval wfdb ni o'rnating.
        """
        import subprocess
        import sys

        url = f"{self.BASE_URL}/{dataset_slug}/"

        # wget rekursiv yuklab oladi
        cmd = [
            "wget",
            "--recursive",
            "--no-parent",
            "--no-host-directories",
            "--cut-dirs=2",
            "--accept=*.edf,*.bdf,*.hea,*.seizures,*.txt",
            f"--directory-prefix={dest_dir}",
            url,
        ]

        if subjects is not None:
            # Subject IDlaridan regex pattern yasaymiz
            pattern = "|".join(subjects)
            cmd.extend(["--accept-regex", pattern])

        logger.info(f"wget buyrug'i ishga tushirilmoqda: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 soat timeout
            )
            if result.returncode != 0:
                logger.warning(f"wget xulosasi (return code {result.returncode}): {result.stderr[:500]}")
            else:
                logger.info("wget muvaffaqiyatli tugadi.")
        except FileNotFoundError:
            raise RuntimeError(
                "wget ham topilmadi. "
                "Iltimos 'pip install wfdb' yoki tizimga wget o'rnating."
            )
        except subprocess.TimeoutExpired:
            logger.error("wget timeout (1 soat). Internetni yoki dataset hajmini tekshiring.")

        return dest_dir

    def _record_exists(self, dest_dir: Path, record: str) -> bool:
        """Record uchun kamida bitta EDF fayl lokal diskda mavjudmi."""
        record_dir = dest_dir / os.path.dirname(record)
        basename = os.path.basename(record)
        return any(
            f.stem == basename
            for f in record_dir.glob("*")
            if f.suffix.lower() in (".edf", ".bdf")
        )

    def _parse_single_chbmit_summary(
        self, summary_path: Path, seizure_map: Dict[str, List[Tuple[float, float]]]
    ) -> None:
        """Bitta CHB-MIT summary faylini parse qiladi va seizure_map ni to'ldiradi."""
        current_file = None
        try:
            with open(summary_path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except OSError as exc:
            logger.warning(f"Summary fayl o'qilmadi: {summary_path}: {exc}")
            return

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("File Name:"):
                current_file = line.split(":", 1)[1].strip()
                if current_file not in seizure_map:
                    seizure_map[current_file] = []

            elif line.startswith("Seizure") and "Start" in line and current_file:
                # "Seizure 1 Start Time: 2996 seconds"
                try:
                    start_sec = float(line.split(":")[-1].replace("seconds", "").strip())
                    # Keyingi qatorda End Time bo'lishi kutiladi
                    if i + 1 < len(lines):
                        end_line = lines[i + 1].strip()
                        if "End" in end_line:
                            end_sec = float(end_line.split(":")[-1].replace("seconds", "").strip())
                            seizure_map[current_file].append((start_sec, end_sec))
                            i += 1  # end qatorini o'tkazib yuboramiz
                except ValueError:
                    pass

            i += 1
