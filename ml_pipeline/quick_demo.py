"""
ml_pipeline/quick_demo.py
============================
TEZKOR DEMO SKRIPT — Internet kerak emas!

Haqiqiy EEG datani yuklab olmasdan, synthetic (demo) signal yaratib
pipeline ni to'liq test qiladi.

Bu skript orqali:
  - Preprocessing pipeline ni test qilish
  - CNN-LSTM model arxitekturasini ko'rish
  - O'qitish loop ni test qilish
  - Baholash va grafiklarni olish

Ishlatilishi::
    python -m ml_pipeline.quick_demo
    # yoki
    python ml_pipeline/quick_demo.py
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

logger = get_logger(__name__, log_file="ml_pipeline/logs/demo.log")


def generate_synthetic_eeg_dataset(
    n_subjects: int = 20,
    n_channels: int = 19,
    sfreq: float = 256.0,
    duration_sec: float = 60.0,
    n_classes: int = 2,
    seed: int = 42,
) -> list:
    """
    Synthetic EEG dataset yaratadi.

    Har bir subject uchun bir nechta to'liq EEG yozuv yaratiladi.
    Normal (class 0) va seizure (class 1) signallar turlicha amplituda
    va ritm xususiyatlariga ega qilinadi.

    Returns:
        recordings: List of recording dicts (BIDSParser kabi format).
    """
    rng = np.random.default_rng(seed)
    recordings: list = []
    times = np.arange(0, duration_sec, 1.0 / sfreq)
    n_samples = len(times)

    for subject_idx in range(n_subjects):
        label = subject_idx % n_classes   # Navbatma-navbat label

        data = np.zeros((n_channels, n_samples), dtype=np.float32)

        for ch_idx in range(n_channels):
            # Asosiy EEG ritmlar
            delta = (2.0 + rng.uniform(-0.5, 0.5)) * np.sin(
                2 * np.pi * 2.0 * times + rng.uniform(0, 2 * np.pi)
            )
            theta = (0.8 + rng.uniform(-0.2, 0.2)) * np.sin(
                2 * np.pi * 6.0 * times + rng.uniform(0, 2 * np.pi)
            )
            alpha = (1.5 - label * 0.6) * np.sin(
                # Seizure paytida alpha kamayadi
                2 * np.pi * 10.0 * times + rng.uniform(0, 2 * np.pi)
            )
            beta = (0.4 + label * 0.3) * np.sin(
                # Seizure paytida beta ortadi
                2 * np.pi * 20.0 * times + rng.uniform(0, 2 * np.pi)
            )
            noise = 0.3 * rng.standard_normal(n_samples)

            channel = delta + theta + alpha + beta + noise

            # Seizure: Ma'lum segmentlarda katta amplitudali spike lar
            if label == 1:
                n_spikes = rng.integers(3, 8)
                for _ in range(n_spikes):
                    spike_center = rng.uniform(5, duration_sec - 5)
                    spike_duration = rng.uniform(0.5, 2.0)
                    spike_mask = (
                        (times >= spike_center) &
                        (times <= spike_center + spike_duration)
                    )
                    # Seizure spike: yuqori amplituda + tez o'zgarish
                    spike_amp = rng.uniform(3.0, 8.0)
                    channel[spike_mask] += (
                        spike_amp * np.sin(2 * np.pi * 3.0 * times[spike_mask])
                    )

            data[ch_idx] = channel * 30.0   # µV ga o'tkazish

        recordings.append({
            "data": data,
            "sfreq": sfreq,
            "channel_names": [
                "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                "T3", "C3", "Cz", "C4", "T4",
                "T5", "P3", "Pz", "P4", "T6",
                "O1", "O2",
            ][:n_channels],
            "source_name": f"synthetic_sub{subject_idx:02d}.edf",
            "source_path": f"data/synthetic/sub{subject_idx:02d}.edf",
            "source_type": "synthetic",
            "subject_id": f"sub-{subject_idx:02d}",
            "task": "seizure_detection",
            "label": label,
            "metadata": {"group": "seizure" if label == 1 else "normal"},
        })

    class_counts = {int(l): sum(1 for r in recordings if r["label"] == l) for l in range(n_classes)}
    logger.info(f"Synthetic dataset: {len(recordings)} subject, taqsimot: {class_counts}")
    return recordings


def run_demo() -> None:
    """To'liq demo pipeline ni ishga tushiradi."""
    logger.info("=" * 60)
    logger.info("  NeuroAnalyst EEG ML Pipeline — QUICK DEMO")
    logger.info("=" * 60)

    set_global_seed(42)

    # ── Konfiguratsiya (kichik demo uchun optimallashtirилган) ─────
    cfg = PipelineConfig(
        data=DataConfig(
            target_sfreq=256.0,
            channels_to_use=None,   # Demo da 19 kanal
        ),
        preprocessing=PreprocessingConfig(
            apply_ica=False,         # Demo uchun tez ishlash uchun o'chirilgan
            epoch_duration_sec=4.0,
            epoch_overlap_sec=2.0,
            epoch_reject_threshold_uv=200.0,
        ),
        features=FeatureConfig(
            use_spectral=True,
            use_temporal=True,
            use_entropy=False,       # Demo uchun sekin ishlaydi
            use_connectivity=False,  # Sekin, demo da o'chirilgan
            apply_pca=False,
        ),
        model=ModelConfig(
            model_type="cnn_lstm",
            n_channels=19,
            n_timepoints=1024,      # 4s × 256Hz
            cnn_filters=[32, 64, 128],  # Kichikroq (tez ishlash uchun)
            lstm_units=64,
            n_classes=2,
            final_dropout=0.4,
        ),
        training=TrainingConfig(
            epochs=20,              # Demo uchun kam epoch
            batch_size=16,
            learning_rate=1e-3,
            early_stopping_patience=5,
            apply_augmentation=True,
            seed=42,
        ),
    )

    output_dir = Path("ml_pipeline/demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(output_dir / "demo_config.json")

    # ── 1. Synthetic data yaratish ────────────────────────────────
    logger.info("\n[1/5] Synthetic EEG dataset yaratilmoqda...")
    t0 = time.time()
    recordings = generate_synthetic_eeg_dataset(
        n_subjects=30,
        n_channels=19,
        sfreq=256.0,
        duration_sec=60.0,
        n_classes=2,
    )
    logger.info(f"  → {len(recordings)} ta yozuv yaratildi ({time.time() - t0:.1f}s)")

    # ── 2. Preprocessing ──────────────────────────────────────────
    logger.info("\n[2/5] Preprocessing...")
    from ml_pipeline.preprocessing.pipeline import PreprocessingPipeline

    t0 = time.time()
    prep_pipeline = PreprocessingPipeline(cfg.data, cfg.preprocessing)
    X, y = prep_pipeline.process_dataset(recordings, fit_normalizer=True)
    logger.info(f"  → X={X.shape}, y={y.shape} ({time.time() - t0:.1f}s)")

    # Subject IDlarini yaratish
    subject_ids = np.array([
        rec["subject_id"]
        for rec in recordings
        for _ in range(max(1, len(X) // len(recordings)))
    ])[:len(X)]

    # ── 3. Train/Val/Test split ────────────────────────────────────
    logger.info("\n[3/5] Subject-wise train/val/test split...")
    from ml_pipeline.training.cross_validation import SubjectWiseCrossValidator

    cv = SubjectWiseCrossValidator(n_splits=5, test_ratio=0.2, seed=42)
    train_idx, val_idx, test_idx = next(cv.split(X, y, subject_ids))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info(
        f"  → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    # ── 4a. CNN-LSTM o'qitish ─────────────────────────────────────
    logger.info("\n[4/5] CNN-LSTM model o'qitilmoqda...")
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

    # ── 4b. Classical ML (XGBoost) ────────────────────────────────
    logger.info("\n[4b/5] XGBoost (feature-based) o'qitilmoqda...")
    from ml_pipeline.features.feature_pipeline import FeaturePipeline
    from ml_pipeline.models.classical_ml import ClassicalEEGClassifier

    feat_pipeline = FeaturePipeline(cfg.features, sfreq=256.0)
    X_train_feats = feat_pipeline.fit_transform(X_train)
    X_val_feats = feat_pipeline.transform(X_val)
    X_test_feats = feat_pipeline.transform(X_test)
    logger.info(f"  Feature matrix: {X_train_feats.shape}")

    # RandomForest (XGBoost o'rnatilmagan bo'lishi mumkin)
    rf_clf = ClassicalEEGClassifier(cfg.model, estimator="random_forest")
    rf_clf.fit(X_train_feats, y_train)

    # ── 5. Baholash ───────────────────────────────────────────────
    logger.info("\n[5/5] Natijalar baholanmoqda...")
    from ml_pipeline.evaluation.metrics import EEGClassificationMetrics
    from ml_pipeline.evaluation.visualizer import EvaluationVisualizer
    from ml_pipeline.evaluation.report_generator import MLReportGenerator

    # DL baholash
    dl_metrics = trainer.evaluate(X_test, y_test)
    y_prob_dl = trainer.predict_proba(X_test)

    # Classical ML baholash
    y_pred_rf = rf_clf.predict(X_test_feats)
    y_prob_rf = rf_clf.predict_proba(X_test_feats)
    metrics_calc = EEGClassificationMetrics()
    rf_metrics = metrics_calc.compute(y_test, y_pred_rf, y_prob_rf)

    # Vizualizatsiya
    viz = EvaluationVisualizer(output_dir=output_dir / "plots")
    viz.plot_training_curves(history, save_name="cnn_lstm_training.png")
    viz.plot_confusion_matrix(y_test, np.argmax(y_prob_dl, axis=1), save_name="cnn_lstm_cm.png")
    viz.plot_roc_curve(y_test, y_prob_dl, save_name="cnn_lstm_roc.png")
    viz.plot_confusion_matrix(y_test, y_pred_rf, save_name="rf_cm.png")

    # Feature importance
    rf_importances = rf_clf.get_feature_importances()
    if rf_importances is not None:
        feat_names = feat_pipeline.get_feature_names()
        viz.plot_feature_importances(rf_importances, feat_names, save_name="rf_feature_importance.png")

    # Hisobotlar
    reporter = MLReportGenerator(output_dir=output_dir / "reports")
    reporter.generate("Synthetic EEG Demo", "CNN-LSTM", dl_metrics, history=history)
    reporter.generate("Synthetic EEG Demo", "Random Forest", rf_metrics)

    # ── Yakuniy xulosa ────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  DEMO MUVAFFAQIYATLI TUGADI!")
    logger.info("=" * 60)
    logger.info(f"  CNN-LSTM Accuracy:     {dl_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  CNN-LSTM Macro F1:     {dl_metrics.get('macro_f1', 0):.4f}")
    logger.info(f"  CNN-LSTM AUROC:        {dl_metrics.get('auroc', 0):.4f}")
    logger.info(f"  RandomForest Accuracy: {rf_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  RandomForest Macro F1: {rf_metrics.get('macro_f1', 0):.4f}")
    logger.info("─" * 60)
    logger.info(f"  Natijalar: {output_dir}")
    logger.info(f"  Grafiklar: {output_dir / 'plots'}")
    logger.info(f"  Hisobotlar: {output_dir / 'reports'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_demo()
