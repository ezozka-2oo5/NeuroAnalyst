"""
ml_pipeline/data_loader/dataset_registry.py
============================================
Qo'llab-quvvatlanadigan EEG datasetlarining markaziy ro'yxati.
Yangi dataset qo'shish uchun faqat shu faylni o'zgartirish yetarli.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetInfo:
    """Bitta dataset haqidagi metadata."""

    # Noyob identifikator (boshqa joyda reference qilish uchun)
    dataset_id: str

    # Inson o'qiy oladigan nom
    name: str

    # Qaysi platformada joylashgan
    source: str   # "physionet" | "openneuro" | "local"

    # PhysioNet content slug yoki OpenNeuro accession ID
    remote_id: str

    # Qaysi kasallik/vazifani o'rganish uchun mo'ljallangan
    task: str

    # Labellash strategiyasi
    label_strategy: str

    # Subjectlar soni (taxminiy)
    n_subjects: int

    # Kanallar soni
    n_channels: int

    # Original sampling rate (Hz)
    original_sfreq: float

    # Taxminiy umumiy hajm (GB)
    size_gb: float

    # Format: "edf" | "bdf" | "bids_edf"
    file_format: str

    # Dataset ichidagi annotation (event) nomlari → label int mapping
    event_to_label: Dict[str, int] = field(default_factory=dict)

    # Qo'shimcha eslatmalar
    notes: Optional[str] = None


# ───────────────────────────── Ro'yxat ─────────────────────────────

DATASET_REGISTRY: Dict[str, DatasetInfo] = {

    # ─── PhysioNet ───────────────────────────────────────────────────

    "chb_mit": DatasetInfo(
        dataset_id="chb_mit",
        name="CHB-MIT Scalp EEG Database",
        source="physionet",
        remote_id="chb-mit-scalp-eeg-database-1.0.0",
        task="seizure_detection",
        label_strategy="annotation_based",
        n_subjects=22,
        n_channels=23,
        original_sfreq=256.0,
        size_gb=46.0,
        file_format="edf",
        event_to_label={"normal": 0, "seizure": 1},
        notes=(
            "Pediatrik epilepsiya bemorlari. "
            "Seizure vaqtlari .seizures faylida yozilgan. "
            "Juda mashhur benchmark dataset."
        ),
    ),

    "eegmmidb": DatasetInfo(
        dataset_id="eegmmidb",
        name="EEG Motor Movement/Imagery Dataset",
        source="physionet",
        remote_id="eeg-motor-movementimagery-dataset-1.0.0",
        task="motor_imagery",
        label_strategy="event_annotation",
        n_subjects=109,
        n_channels=64,
        original_sfreq=160.0,
        size_gb=3.0,
        file_format="edf",
        event_to_label={
            "T0": 0,   # Rest
            "T1": 1,   # Sol qo'l harakat
            "T2": 2,   # O'ng qo'l harakat
        },
        notes=(
            "BCI uchun keng ishlatiladigan dataset. "
            "64 kanal, 109 subject, har xil motor vazifalar."
        ),
    ),

    "sleep_edf": DatasetInfo(
        dataset_id="sleep_edf",
        name="Sleep-EDF Database (Expanded)",
        source="physionet",
        remote_id="sleep-edfx-database-1.0.0",
        task="sleep_staging",
        label_strategy="annotation_based",
        n_subjects=153,
        n_channels=2,
        original_sfreq=100.0,
        size_gb=6.8,
        file_format="edf",
        event_to_label={
            "Sleep stage W":  0,   # Uyg'oqlik
            "Sleep stage 1":  1,   # NREM 1
            "Sleep stage 2":  2,   # NREM 2
            "Sleep stage 3":  3,   # NREM 3
            "Sleep stage 4":  4,   # NREM 4
            "Sleep stage R":  5,   # REM
        },
        notes=(
            "Uyqu stadiyalarini tasniflash uchun standart dataset. "
            "Faqat 2 kanal (Fpz-Cz, Pz-Oz) lekin yaxshi annotatsiyalangan."
        ),
    ),

    "tuh_eeg": DatasetInfo(
        dataset_id="tuh_eeg",
        name="Temple University Hospital EEG Corpus",
        source="physionet",
        remote_id="tuh-eeg-corpus",
        task="abnormal_detection",
        label_strategy="report_based",
        n_subjects=10000,
        n_channels=22,
        original_sfreq=250.0,
        size_gb=1500.0,
        file_format="edf",
        event_to_label={"normal": 0, "abnormal": 1},
        notes=(
            "Dunyodagi eng katta klinik EEG to'plami (TUH). "
            "Subsetlar: TUH Seizure, TUH Artifact, TUH Slowing. "
            "Ro'yxatdan o'tish talab etiladi: https://isip.piconepress.com/projects/tuh_eeg/"
        ),
    ),

    "tuh_seizure": DatasetInfo(
        dataset_id="tuh_seizure",
        name="TUH EEG Seizure Corpus",
        source="physionet",
        remote_id="tuh-eeg-seizure",
        task="seizure_detection",
        label_strategy="annotation_based",
        n_subjects=675,
        n_channels=22,
        original_sfreq=250.0,
        size_gb=100.0,
        file_format="edf",
        event_to_label={
            "bckg": 0,   # Background (normal)
            "seiz": 1,   # Seizure
            "fnsz": 1,   # Focal non-specific seizure
            "gnsz": 1,   # General non-specific seizure
            "tcsz": 1,   # Tonic-clonic seizure
        },
        notes="TUH ning faqat seizure annotatsiyalangan subkolleksiyasi.",
    ),

    "siena": DatasetInfo(
        dataset_id="siena",
        name="Siena Scalp EEG Database",
        source="physionet",
        remote_id="siena-scalp-eeg-database-1.0.0",
        task="seizure_detection",
        label_strategy="annotation_based",
        n_subjects=14,
        n_channels=29,
        original_sfreq=512.0,
        size_gb=14.0,
        file_format="edf",
        event_to_label={"normal": 0, "seizure": 1},
        notes="Italiya (Siena) klinikasi ma'lumotlari. Yuqori sfreq.",
    ),

    # ─── OpenNeuro (BIDS formatida) ──────────────────────────────────

    "ds002778": DatasetInfo(
        dataset_id="ds002778",
        name="A multi-subject, multi-modal human neuroimaging dataset",
        source="openneuro",
        remote_id="ds002778",
        task="resting_state",
        label_strategy="metadata",
        n_subjects=204,
        n_channels=64,
        original_sfreq=1000.0,
        size_gb=8.0,
        file_format="bids_edf",
        event_to_label={"rest": 0},
        notes="BIDS formati. EEG resting-state. OpenNeuro tavsiya qilgan.",
    ),

    "ds003490": DatasetInfo(
        dataset_id="ds003490",
        name="EEG dataset of visual steady-state evoked potentials (SSVEP)",
        source="openneuro",
        remote_id="ds003490",
        task="ssvep",
        label_strategy="event_annotation",
        n_subjects=30,
        n_channels=64,
        original_sfreq=256.0,
        size_gb=2.1,
        file_format="bids_edf",
        event_to_label={"6hz": 0, "10hz": 1, "15hz": 2},
        notes="Brain-Computer Interface SSVEP paradigm.",
    ),

    "ds004504": DatasetInfo(
        dataset_id="ds004504",
        name="EEG Dataset of Alzheimer's and Frontotemporal Dementia",
        source="openneuro",
        remote_id="ds004504",
        task="dementia_classification",
        label_strategy="metadata",
        n_subjects=88,
        n_channels=19,
        original_sfreq=500.0,
        size_gb=1.8,
        file_format="bids_edf",
        event_to_label={"A": 0, "F": 1, "C": 2},  # Alzheimer, FTD, Control
        notes=(
            "Alzheimer va Frontotemporal Demensiya klassifikatsiyasi. "
            "19 kanal — xalqaro 10-20 standart. Klinik jihatdan muhim."
        ),
    ),
}


class DatasetRegistry:
    """Dataset ro'yxatiga qulay kirish interfeysini ta'minlovchi klass."""

    @staticmethod
    def get(dataset_id: str) -> DatasetInfo:
        """
        Dataset ID bo'yicha ma'lumot olish.

        Args:
            dataset_id: Registry dagi kalit (masalan "chb_mit").

        Returns:
            DatasetInfo instance.

        Raises:
            KeyError: Noto'g'ri ID berilsa.
        """
        if dataset_id not in DATASET_REGISTRY:
            available = ", ".join(DATASET_REGISTRY.keys())
            raise KeyError(
                f"Dataset '{dataset_id}' topilmadi. "
                f"Mavjud datasetlar: {available}"
            )
        return DATASET_REGISTRY[dataset_id]

    @staticmethod
    def list_by_task(task: str) -> List[DatasetInfo]:
        """Biror vazifaga mos keladigan datasetlarni ro'yxati."""
        return [info for info in DATASET_REGISTRY.values() if info.task == task]

    @staticmethod
    def list_by_source(source: str) -> List[DatasetInfo]:
        """Biror platformadagi datasetlar ro'yxati."""
        return [info for info in DATASET_REGISTRY.values() if info.source == source]

    @staticmethod
    def all_ids() -> List[str]:
        """Barcha mavjud dataset IDlarini qaytaradi."""
        return list(DATASET_REGISTRY.keys())

    @staticmethod
    def summary() -> None:
        """Barcha datasetlarning qisqa xulosa jadvalini chiqaradi."""
        header = f"{'ID':<20} {'Task':<25} {'Source':<12} {'Subjects':>8} {'GB':>6}"
        print(header)
        print("-" * len(header))
        for info in DATASET_REGISTRY.values():
            print(
                f"{info.dataset_id:<20} {info.task:<25} {info.source:<12} "
                f"{info.n_subjects:>8} {info.size_gb:>6.1f}"
            )
