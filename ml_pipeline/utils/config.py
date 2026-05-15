"""
ml_pipeline/utils/config.py
============================
Global konfiguratsiya klassi.
Barcha pipeline parametrlari shu yerda markazlashtirilgan — tarqoq magic numberlardan qochish uchun.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    """Ma'lumotlarni yuklash uchun konfiguratsiya."""

    # PhysioNet dataset nomi (physionet.org dagi content slug)
    physionet_dataset: str = "eegmmidb"   # EEG Motor Movement/Imagery Dataset

    # OpenNeuro BIDS dataset accession ID
    openneuro_dataset_id: str = "ds002778"

    # Yuklab olinadigan fayllarning saqlash joyi
    raw_data_dir: Path = Path("data/raw")

    # Preprocessing natijalari saqlanadigan joy
    processed_data_dir: Path = Path("data/processed")

    # Cached epoch/numpy fayllar uchun
    cache_dir: Path = Path("data/cache")

    # Ko'rsatilgan kanallar ro'yxati (10-20 standart, None = hammasi)
    channels_to_use: Optional[List[str]] = field(
        default_factory=lambda: [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T3", "C3", "Cz", "C4", "T4",
            "T5", "P3", "Pz", "P4", "T6",
            "O1", "O2",
        ]
    )

    # Maqsadli sampling chastotasi (Hz) — resampling qilinadi
    target_sfreq: float = 256.0

    # Teglar xaritasi: dataset ichidagi event annotation → bizning label integer
    label_map: Dict[str, int] = field(
        default_factory=lambda: {
            "normal": 0,
            "seizure": 1,
            "artifact": 2,
        }
    )

    def __post_init__(self):
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)
        self.cache_dir = Path(self.cache_dir)


@dataclass
class PreprocessingConfig:
    """Signal tozalash konfiguratsiyasi."""

    # Bandpass filter chegaralari (Hz)
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    bandpass_order: int = 5

    # Notch filter — elektr tarmog'i chastotasi
    notch_hz: float = 50.0
    notch_q: float = 30.0          # Q-factor: qanchalik tor kessini belgilaydi

    # Re-reference usuli: "average" | "linked_mastoids" | None
    reference: Optional[str] = "average"

    # ICA artifact removal
    apply_ica: bool = True
    ica_n_components: int = 20     # PCA komponentlari soni
    ica_method: str = "fastica"    # "fastica" | "picard" | "infomax"
    ica_max_iter: int = 500

    # Kanallarni rad qilish chegarasi (µV standart og'ish bo'yicha)
    flat_channel_threshold_uv: float = 3.0    # std(ch) < 3µV → flat channel
    noisy_channel_threshold_z: float = 4.0   # z-score > 4 → noisy channel

    # Epochlash
    epoch_duration_sec: float = 4.0       # Har bir epoch uzunligi
    epoch_overlap_sec: float = 2.0        # Overlap — augmentatsiya uchun
    baseline_correction: bool = True      # Baseline (-0.2→0) subtraction
    baseline_window: Tuple[float, float] = (-0.2, 0.0)

    # Yomon epochlarni tashlab yuborish chegarasi (µV amplitude)
    epoch_reject_threshold_uv: float = 150.0


@dataclass
class FeatureConfig:
    """Feature extraction sozlamalari."""

    # Qaysi feature turlarini hisoblash
    use_spectral: bool = True       # PSD-based band powers
    use_temporal: bool = True       # Statistical moments, Hjorth params
    use_connectivity: bool = False  # Coherence — qo'shimcha hisoblash kerak

    # PSD Welch usuli parametrlari
    welch_nperseg: int = 512
    welch_noverlap: int = 256

    # EEG ritmlar (Hz)
    bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "delta":  (0.5,  4.0),
            "theta":  (4.0,  8.0),
            "alpha":  (8.0, 12.0),
            "beta":  (12.0, 30.0),
            "gamma": (30.0, 45.0),
        }
    )

    # Temporal features
    use_hjorth: bool = True        # Activity, Mobility, Complexity
    use_entropy: bool = True       # Approximate Entropy, Sample Entropy
    use_statistics: bool = True    # Mean, Std, Skewness, Kurtosis, p2p

    # Feature scaling usuli: "standard" | "minmax" | "robust"
    scaler_type: str = "robust"

    # PCA yordamida o'lchamni kamaytirish
    apply_pca: bool = False
    pca_variance_ratio: float = 0.95   # 95% dispersion saqlansin


@dataclass
class ModelConfig:
    """Model arxitekturasi sozlamalari."""

    # Qaysi modelni o'qitish: "cnn1d" | "cnn_lstm" | "conformer" | "classical"
    model_type: str = "cnn_lstm"

    # Input dimensions (epoch_duration * sfreq namunalar, kanal soni)
    n_channels: int = 19
    n_timepoints: int = 1024       # 4 soniya × 256 Hz

    # CNN qatlam sozlamalari (cnn1d va cnn_lstm uchun)
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_size: int = 25      # 25 namuna ≈ ~100ms at 256Hz
    cnn_stride: int = 1
    cnn_pool_size: int = 4
    cnn_dropout: float = 0.25

    # LSTM qatlam sozlamalari (cnn_lstm uchun)
    lstm_units: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True

    # Transformer/Conformer sozlamalari
    conformer_d_model: int = 256
    conformer_n_heads: int = 8
    conformer_n_layers: int = 6
    conformer_ff_dim: int = 1024
    conformer_dropout: float = 0.1

    # Chiqish
    n_classes: int = 2
    final_dropout: float = 0.5

    # Classical ML sozlamalari
    classical_estimator: str = "xgboost"  # "xgboost" | "random_forest" | "svm"
    classical_cv_folds: int = 5


@dataclass
class TrainingConfig:
    """O'qitish jarayoni sozlamalari."""

    # O'qitish parametrlari
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4           # L2 regularization

    # Optimizer: "adam" | "adamw" | "sgd"
    optimizer: str = "adamw"

    # LR Scheduler: "cosine" | "step" | "plateau" | None
    scheduler: Optional[str] = "cosine"
    scheduler_warmup_epochs: int = 5
    scheduler_min_lr: float = 1e-6

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_f1"  # monitor nima qilinsin
    early_stopping_mode: str = "max"

    # Cross-validation
    cv_strategy: str = "subject_wise"     # "subject_wise" | "kfold" | "stratified"
    n_splits: int = 5
    test_subject_ratio: float = 0.2       # Testga ajratiladigan subjectlar ulushi

    # Data augmentation
    apply_augmentation: bool = True
    aug_noise_std: float = 0.05           # Gaussian noise shtampi
    aug_time_shift_max: int = 50          # Namunalarda vaqtni siljitish
    aug_amplitude_scale: Tuple[float, float] = (0.9, 1.1)

    # Class imbalance
    use_class_weights: bool = True
    oversample_minority: bool = False     # SMOTE-like oversampling

    # Checkpointing
    checkpoint_dir: Path = Path("ml_pipeline/checkpoints")
    save_best_only: bool = True
    checkpoint_metric: str = "val_f1"

    # Logging
    log_dir: Path = Path("ml_pipeline/logs")
    log_every_n_steps: int = 10

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)


@dataclass
class PipelineConfig:
    """
    Barcha konfiguratsiyalarni birlashtiruvchi asosiy klass.

    Ishlatilishi::

        cfg = PipelineConfig()
        cfg.save("config.json")
        cfg2 = PipelineConfig.load("config.json")
    """

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path: str | Path) -> None:
        """Konfiguratsiyani JSON formatda diskka yozish."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, default=_convert)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        """JSON fayldan konfiguratsiyani yuklash."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        # JSON da tuple lar list sifatida saqlanadi — qayta aylantirish
        prep_raw = raw["preprocessing"]
        if "baseline_window" in prep_raw and isinstance(prep_raw["baseline_window"], list):
            prep_raw["baseline_window"] = tuple(prep_raw["baseline_window"])

        train_raw = raw["training"]
        if "aug_amplitude_scale" in train_raw and isinstance(train_raw["aug_amplitude_scale"], list):
            train_raw["aug_amplitude_scale"] = tuple(train_raw["aug_amplitude_scale"])

        # features.bands dict ichidagi listlarni tuple ga aylantirish
        feat_raw = raw["features"]
        if "bands" in feat_raw and isinstance(feat_raw["bands"], dict):
            feat_raw["bands"] = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in feat_raw["bands"].items()
            }

        cfg = cls(
            data=DataConfig(**raw["data"]),
            preprocessing=PreprocessingConfig(**prep_raw),
            features=FeatureConfig(**feat_raw),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**train_raw),
        )
        return cfg

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """
        Muhit o'zgaruvchilaridan (environment variables) konfiguratsiya yaratish.
        Bu production deploymentlarda qulay.
        """
        cfg = cls()
        if raw_dir := os.environ.get("NEURO_RAW_DATA_DIR"):
            cfg.data.raw_data_dir = Path(raw_dir)
        if model_type := os.environ.get("NEURO_MODEL_TYPE"):
            cfg.model.model_type = model_type
        if seed := os.environ.get("NEURO_SEED"):
            cfg.training.seed = int(seed)
        if sfreq := os.environ.get("NEURO_TARGET_SFREQ"):
            cfg.data.target_sfreq = float(sfreq)
        return cfg
