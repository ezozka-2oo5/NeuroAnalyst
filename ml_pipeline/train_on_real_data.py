"""
ml_pipeline/train_on_real_data.py
===================================
OpenNeuro ds004504 datasetini yuklab olib, haqiqiy bemor ma'lumotlarida
CNN-LSTM modelini o'qitish skripti.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ml_pipeline.utils.config import (
    DataConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    PreprocessingConfig,
    TrainingConfig,
)
from ml_pipeline.utils.logger import get_logger
from ml_pipeline.utils.seed import set_global_seed
from ml_pipeline.data_loader.dataset_registry import DatasetRegistry
from ml_pipeline.data_loader.openneuro_loader import OpenNeuroLoader
from ml_pipeline.data_loader.bids_parser import BIDSParser

logger = get_logger(__name__, log_file="ml_pipeline/logs/real_training.log")


def main():
    logger.info("=" * 60)
    logger.info("  NeuroAnalyst - Haqiqiy EEG Ma'lumotlarda O'qitish (ds004504)")
    logger.info("=" * 60)

    set_global_seed(42)

    # ── 1. Konfiguratsiya ─────────────────────────────────────────
    # ds004504: 19 kanal, 500Hz. Biz uni 250Hz ga tushiramiz (tezlik uchun).
    # 3 ta sinf: 0=Alzheimer, 1=FTD, 2=Control
    cfg = PipelineConfig(
        data=DataConfig(
            target_sfreq=250.0,
            raw_data_dir="data/raw_data",
        ),
        preprocessing=PreprocessingConfig(
            apply_ica=False,  # Tezlashtirish uchun o'chirilgan
            epoch_duration_sec=4.0,
            epoch_overlap_sec=2.0,
            epoch_reject_threshold_uv=200.0,
        ),
        features=FeatureConfig(
            use_spectral=True,
            use_temporal=True,
            use_entropy=False,  # Sekin
        ),
        model=ModelConfig(
            model_type="cnn_lstm",
            n_channels=19,
            n_timepoints=1000,   # 4s × 250Hz = 1000
            cnn_filters=[32, 64, 128],
            lstm_units=64,
            n_classes=3,         # A, F, C
        ),
        training=TrainingConfig(
            epochs=20,
            batch_size=16,
            learning_rate=1e-3,
            early_stopping_patience=5,
        ),
    )

    output_dir = Path("ml_pipeline/real_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(output_dir / "real_config.json")

    # ── 2. Data Yuklab olish (OpenNeuro) ──────────────────────────
    dataset_info = DatasetRegistry.get("ds004504")
    loader = OpenNeuroLoader(cfg.data)
    
    logger.info(f"\n[1/5] OpenNeuro dan dataset yuklanmoqda: {dataset_info.name}...")
    t0 = time.time()
    
    # Tez ishlashi uchun max_subjects=6 (har xil guruhlardan)
    # BIDS formatida sub-001 kabi yozilishi mumkin
    # Biz A, C, F klasslaridan 2 tadan bemor tanlaymiz:
    selected_subjects = ["sub-001", "sub-002", "sub-037", "sub-038", "sub-066", "sub-067"]
    download_dir = loader.download(
        "ds004504", 
        subjects=selected_subjects, 
        max_subjects=None, 
        force_redownload=False
    )
    logger.info(f"  → Yuklash tugadi ({time.time() - t0:.1f}s). Manzil: {download_dir}")

    # ── 3. BIDS formatini Parse qilish ──────────────────────────
    logger.info("\n[2/5] BIDS metadata va EDF fayllarini parse qilish...")
    bids = BIDSParser(cfg.data)
    
    # subjects orqali faqat tanlanganlarni yuklaymiz (bids_parser prefixsiz kutadi)
    subject_ids = [s.replace("sub-", "") for s in selected_subjects]
    recordings = bids.load_dataset(
        download_dir, 
        label_map=dataset_info.event_to_label,
        subjects=subject_ids,
    )
    
    if len(recordings) == 0:
        logger.error("Hech qanday recording topilmadi. Yuklash xato yakunlangan bo'lishi mumkin.")
        return

    # ── 4. Preprocessing ──────────────────────────────────────────
    logger.info("\n[3/5] Preprocessing (Tozalash va Epoch qilish)...")
    from ml_pipeline.preprocessing.pipeline import PreprocessingPipeline
    prep_pipeline = PreprocessingPipeline(cfg.data, cfg.preprocessing)
    
    t0 = time.time()
    X, y = prep_pipeline.process_dataset(recordings, fit_normalizer=True)
    logger.info(f"  → X={X.shape}, y={y.shape} ({time.time() - t0:.1f}s)")
    
    subject_ids = np.array([
        rec.get("subject_id", "sub-unknown") 
        for rec in recordings 
        for _ in range(max(1, len(X) // len(recordings)))
    ])[:len(X)]

    # ── 5. Train/Val/Test bo'lish (Subject-wise) ─────────────────
    logger.info("\n[4/5] Subject-wise split...")
    from ml_pipeline.training.cross_validation import SubjectWiseCrossValidator
    
    cv = SubjectWiseCrossValidator(n_splits=3, test_ratio=0.2, seed=42)
    try:
        train_idx, val_idx, test_idx = next(cv.split(X, y, subject_ids))
    except ValueError as e:
        # Agar subjectlar juda kam bo'lsa
        logger.warning(f"Subject-wise split amalga oshmadi: {e}. Random split qilinmoqda.")
        from sklearn.model_selection import train_test_split
        train_val_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info(f"  → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── 6. O'qitish (CNN-LSTM) ────────────────────────────────────
    logger.info("\n[5/5] CNN-LSTM modelni haqiqiy datada o'qitish...")
    from ml_pipeline.models.cnn_lstm import EEGCNNLSTM
    from ml_pipeline.training.trainer import EEGModelTrainer

    model = EEGCNNLSTM(cfg.model)
    trainer = EEGModelTrainer(
        cfg.training, model,
        checkpoint_dir=output_dir / "checkpoints",
        log_dir=output_dir / "logs",
    )

    t0 = time.time()
    history = trainer.fit(X_train, y_train, X_val, y_val)
    logger.info(f"  → O'qitish tugadi ({time.time() - t0:.1f}s)")

    # ── 7. Baholash ──────────────────────────────────────────────
    if len(X_test) > 0:
        logger.info("\n[*] Model test qilinmoqda...")
        dl_metrics = trainer.evaluate(X_test, y_test)
        
        logger.info(f"  Test Accuracy: {dl_metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Test Macro F1: {dl_metrics.get('macro_f1', 0):.4f}")
    else:
        logger.info("\n[*] Test qismi uchun yetarli subject bo'lmadi.")

    logger.info("\n" + "=" * 60)
    logger.info("  HAQIQIY MA'LUMOTDA O'QITISH MUVAFFAQIYATLI YAKUNLANDI!")
    logger.info(f"  Modellar va grafiklar joylashgan papka: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
