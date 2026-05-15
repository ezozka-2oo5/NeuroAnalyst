"""
ml_pipeline/run_pipeline.py
==============================
ASOSIY ISHGA TUSHIRISH SKRIPTI

Ushbu fayl butun EEG ML pipeline ni boshqaradi:
1. Dataset yuklab olish (PhysioNet / OpenNeuro)
2. Preprocessing (filtrlash, epochlash, normalizatsiya)
3. Feature extraction (spectral + temporal)
4. Model o'qitish (CNN-LSTM yoki Classical ML)
5. Baholash va hisobot

Ishlatilishi::

    # CHB-MIT dataset bilan CNN-LSTM o'qitish
    python -m ml_pipeline.run_pipeline --dataset chb_mit --model cnn_lstm

    # EEG Motor Imagery bilan XGBoost (feature-based)
    python -m ml_pipeline.run_pipeline --dataset eegmmidb --model xgboost

    # Konfiguratsiyani JSON dan yuklab o'qitish
    python -m ml_pipeline.run_pipeline --config my_config.json

    # Faqat kichik sinov uchun (10 fayl)
    python -m ml_pipeline.run_pipeline --dataset chb_mit --model cnn_lstm --max_files 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Pipeline modullarini import qilish
from ml_pipeline.data_loader.dataset_registry import DatasetRegistry
from ml_pipeline.data_loader.physionet_loader import PhysioNetLoader
from ml_pipeline.data_loader.openneuro_loader import OpenNeuroLoader
from ml_pipeline.data_loader.bids_parser import BIDSParser
from ml_pipeline.preprocessing.pipeline import PreprocessingPipeline
from ml_pipeline.features.feature_pipeline import FeaturePipeline
from ml_pipeline.models.cnn_lstm import EEGCNNLSTM
from ml_pipeline.models.cnn_1d import EEG1DCNN
from ml_pipeline.models.eeg_conformer import EEGConformer
from ml_pipeline.models.classical_ml import ClassicalEEGClassifier
from ml_pipeline.training.trainer import EEGModelTrainer
from ml_pipeline.training.cross_validation import SubjectWiseCrossValidator
from ml_pipeline.evaluation.metrics import EEGClassificationMetrics
from ml_pipeline.evaluation.visualizer import EvaluationVisualizer
from ml_pipeline.evaluation.report_generator import MLReportGenerator
from ml_pipeline.utils.config import (
    DataConfig,
    ModelConfig,
    PipelineConfig,
    PreprocessingConfig,
    TrainingConfig,
    FeatureConfig,
)
from ml_pipeline.utils.logger import get_logger
from ml_pipeline.utils.seed import set_global_seed

import numpy as np

logger = get_logger(__name__, log_file="ml_pipeline/logs/pipeline.log")


# ──────────────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroAnalyst EEG ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset tanlov
    parser.add_argument(
        "--dataset",
        type=str,
        default="chb_mit",
        choices=DatasetRegistry.all_ids(),
        help="O'qitiladigan EEG dataset ID (default: chb_mit)",
    )

    # Model tanlov
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_lstm",
        choices=["cnn_lstm", "cnn1d", "conformer", "xgboost", "random_forest", "svm"],
        help="Model arxitekturasi (default: cnn_lstm)",
    )

    # Konfiguratsiya
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON konfiguratsiya fayl yo'li (berilsa, boshqa argumentlar e'tiborga olinmaydi)",
    )

    # Ma'lumot joylari
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Ma'lumotlar saqlash katalogi",
    )

    # Download
    parser.add_argument(
        "--download",
        action="store_true",
        help="Dataset yuklab olsin (mavjud bo'lmasa talab qilinadi)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Debug uchun N ta fayl bilan cheklash",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="Faqat ma'lum subjectlar (masalan: chb01 chb02)",
    )

    # O'qitish
    parser.add_argument("--epochs", type=int, default=100, help="O'qitish epochlari soni")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch o'lchami")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Chiqish
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ml_pipeline",
        help="Checkpoints, loglar va grafiklar uchun katalog",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Dataset yuklab olishni o'tkazib yuborish (lokal mavjud bo'lsa)",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="O'qitishni o'tkazib, faqat baholash qilish",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Subject-wise cross-validation bajarish",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────
# Pipeline bosqichlari
# ──────────────────────────────────────────────────────────────────

def step_download(dataset_info, cfg_data: DataConfig, args) -> Path:
    """
    1-bosqich: Dataset yuklab olish.
    """
    logger.info(f"=== BOSQICH 1: Dataset yuklab olish ({dataset_info.name}) ===")

    if dataset_info.source == "physionet":
        loader = PhysioNetLoader(cfg_data)
        dest = loader.download(
            dataset_info.remote_id,
            subjects=args.subjects,
            max_files=args.max_files,
        )
    elif dataset_info.source == "openneuro":
        loader = OpenNeuroLoader(cfg_data)
        dest = loader.download(
            dataset_info.remote_id,
            subjects=args.subjects,
            max_subjects=args.max_files,
        )
    else:
        dest = Path(cfg_data.raw_data_dir) / "local" / dataset_info.dataset_id
        logger.info(f"Lokal dataset: {dest}")

    return dest


def step_load_recordings(dataset_info, cfg_data: DataConfig, data_dir: Path) -> list:
    """
    2-bosqich: EDF fayllarini yuklash va parsing.
    """
    logger.info("=== BOSQICH 2: EDF fayllarini yuklash ===")

    parser = BIDSParser(cfg_data)

    if dataset_info.source == "openneuro":
        # BIDS format — to'liq metadata bilan
        loader = OpenNeuroLoader(cfg_data)
        recordings = parser.load_dataset(
            data_dir,
            label_map=dataset_info.event_to_label,
        )
    else:
        # PhysioNet — EDF fayllarni alohida yuklash
        physio_loader = PhysioNetLoader(cfg_data)
        edf_paths = physio_loader.get_local_edf_paths(dataset_info.remote_id)

        if not edf_paths:
            raise FileNotFoundError(
                f"EDF fayllar topilmadi. --download flag bilan qayta ishga tushiring."
            )

        recordings = []
        for edf_path in edf_paths:
            rec = parser.load_edf(edf_path)
            if rec:
                recordings.append(rec)

        logger.info(f"{len(recordings)} ta yozuv yuklandi.")

    return recordings


def step_preprocessing(
    recordings: list,
    dataset_info,
    physio_loader: "PhysioNetLoader | None",
    cfg_data: DataConfig,
    cfg_prep: PreprocessingConfig,
) -> tuple:
    """
    3-bosqich: Preprocessing — filtrlash, epochlash, normalizatsiya.
    """
    logger.info("=== BOSQICH 3: Preprocessing ===")

    pipeline = PreprocessingPipeline(cfg_data, cfg_prep)

    # CHB-MIT uchun seizure annotatsiyalari
    seizure_map = None
    if dataset_info.dataset_id == "chb_mit" and physio_loader:
        seizure_map = physio_loader.parse_chbmit_seizure_annotations(dataset_info.remote_id)
        logger.info(f"Seizure annotatsiyalari: {len(seizure_map)} fayl")

    X, y = pipeline.process_dataset(
        recordings,
        seizure_map=seizure_map,
        fit_normalizer=True,
    )

    # Subject IDlarini yaratish (cross-validation uchun)
    subject_ids = np.array([
        rec.get("subject_id", f"sub_{i}")
        for i, rec in enumerate(recordings)
        for _ in range(max(1, len(X) // max(len(recordings), 1)))
    ])[:len(X)]

    logger.info(f"Preprocessing tugadi: X={X.shape}, y={y.shape}")
    logger.info(f"Label taqsimot: { {int(l): int(np.sum(y == l)) for l in np.unique(y)} }")

    return X, y, subject_ids, pipeline


def step_train_deep_learning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg_model: ModelConfig,
    cfg_train: TrainingConfig,
    output_dir: Path,
) -> tuple:
    """
    4a-bosqich: Deep Learning model o'qitish.
    """
    logger.info(f"=== BOSQICH 4: {cfg_model.model_type.upper()} o'qitish ===")

    # Model yaratish
    model_map = {
        "cnn_lstm": EEGCNNLSTM,
        "cnn1d": EEG1DCNN,
        "conformer": EEGConformer,
    }

    model_cls = model_map.get(cfg_model.model_type)
    if model_cls is None:
        raise ValueError(f"Noma'lum DL model: {cfg_model.model_type}")

    model = model_cls(cfg_model)

    trainer = EEGModelTrainer(
        cfg_train,
        model,
        checkpoint_dir=output_dir / "checkpoints",
        log_dir=output_dir / "logs",
    )

    history = trainer.fit(X_train, y_train, X_val, y_val)
    return trainer, history


def step_train_classical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg_model: ModelConfig,
    cfg_features: FeatureConfig,
    sfreq: float,
    output_dir: Path,
) -> tuple:
    """
    4b-bosqich: Classical ML model o'qitish (feature-based).
    """
    logger.info(f"=== BOSQICH 4: {cfg_model.classical_estimator.upper()} o'qitish ===")

    # Feature extraction
    feat_pipeline = FeaturePipeline(cfg_features, sfreq=sfreq)
    X_train_feats = feat_pipeline.fit_transform(X_train)
    X_val_feats = feat_pipeline.transform(X_val)

    logger.info(f"Feature matrix: {X_train_feats.shape}")

    # Model yaratish va o'qitish
    clf = ClassicalEEGClassifier(cfg_model, estimator=cfg_model.classical_estimator)
    clf.fit(X_train_feats, y_train, X_val=X_val_feats, y_val=y_val)

    # Feature pipeline saqlash
    feat_pipeline.save(output_dir / "feature_pipeline.pkl")
    clf.save(output_dir / f"{cfg_model.classical_estimator}_model.pkl")

    return clf, feat_pipeline


def step_evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    is_deep_learning: bool,
    feat_pipeline=None,
    dataset_name: str = "Unknown",
    model_name: str = "Unknown",
    history=None,
    output_dir: Path = Path("ml_pipeline"),
) -> dict:
    """
    5-bosqich: Test to'plamida baholash va hisobot.
    """
    logger.info("=== BOSQICH 5: Baholash ===")

    if is_deep_learning:
        metrics = model.evaluate(X_test, y_test)
        y_prob = model.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1)
    else:
        X_test_feats = feat_pipeline.transform(X_test)
        y_pred = model.predict(X_test_feats)
        y_prob = model.predict_proba(X_test_feats)
        metrics_calc = EEGClassificationMetrics()
        metrics = metrics_calc.compute(y_test, y_pred, y_prob)
        metrics_calc.print_report(metrics)

    # Vizualizatsiya
    viz = EvaluationVisualizer(output_dir=output_dir / "plots")

    if history:
        viz.plot_training_curves(history)

    viz.plot_confusion_matrix(y_test, y_pred)
    viz.plot_roc_curve(y_test, y_prob)

    # Feature importances (classical ML uchun)
    if not is_deep_learning and hasattr(model, "get_feature_importances"):
        imps = model.get_feature_importances()
        if imps is not None and feat_pipeline:
            feat_names = feat_pipeline.get_feature_names()
            viz.plot_feature_importances(imps, feat_names)

    # Hisobot
    reporter = MLReportGenerator(output_dir=output_dir / "reports")
    reporter.generate(
        dataset_name=dataset_name,
        model_name=model_name,
        metrics=metrics,
        history=history,
    )

    return metrics


# ──────────────────────────────────────────────────────────────────
# Asosiy funksiya
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    # ── Konfiguratsiya ────────────────────────────────────────────
    if args.config:
        logger.info(f"Konfiguratsiya yuklanmoqda: {args.config}")
        cfg = PipelineConfig.load(args.config)
    else:
        cfg = PipelineConfig(
            data=DataConfig(raw_data_dir=args.data_dir),
            preprocessing=PreprocessingConfig(),
            features=FeatureConfig(),
            model=ModelConfig(
                model_type=args.model if args.model not in ("xgboost", "random_forest", "svm") else "cnn_lstm",
                classical_estimator=args.model if args.model in ("xgboost", "random_forest", "svm") else "xgboost",
                n_classes=2,
            ),
            training=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=args.seed,
            ),
        )
        # Konfiguratsiyani saqlash (keyinchalik qayta ishlatish uchun)
        cfg.save(f"{args.output_dir}/config.json")
        logger.info(f"Konfiguratsiya saqlandi: {args.output_dir}/config.json")

    output_dir = Path(args.output_dir)
    dataset_info = DatasetRegistry.get(args.dataset)
    is_classical = args.model in ("xgboost", "random_forest", "svm")

    logger.info("=" * 60)
    logger.info(f"  NeuroAnalyst EEG ML Pipeline")
    logger.info(f"  Dataset: {dataset_info.name}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Subjectlar: {args.subjects or 'hammasi'}")
    logger.info("=" * 60)

    # ── 1. Download ───────────────────────────────────────────────
    if not args.skip_download:
        data_dir = step_download(dataset_info, cfg.data, args)
    else:
        if dataset_info.source == "physionet":
            data_dir = Path(cfg.data.raw_data_dir) / "physionet" / dataset_info.remote_id
        else:
            data_dir = Path(cfg.data.raw_data_dir) / "openneuro" / dataset_info.remote_id

    # ── 2. Yuklash ────────────────────────────────────────────────
    recordings = step_load_recordings(dataset_info, cfg.data, data_dir)

    if args.max_files:
        recordings = recordings[:args.max_files]
        logger.info(f"Cheklandi: {len(recordings)} ta yozuv")

    # ── 3. Preprocessing ──────────────────────────────────────────
    physio_loader = PhysioNetLoader(cfg.data) if dataset_info.source == "physionet" else None
    X, y, subject_ids, prep_pipeline = step_preprocessing(
        recordings, dataset_info, physio_loader, cfg.data, cfg.preprocessing
    )

    if len(X) == 0:
        logger.error("Preprocessing dan keyin epoch qolmadi. Pipeline to'xtatildi.")
        sys.exit(1)

    # ── 4. Cross-validation yoki oddiy split ──────────────────────
    if args.cv:
        logger.info("Subject-wise Cross-Validation rejimi.")
        cv = SubjectWiseCrossValidator(
            n_splits=cfg.training.n_splits,
            test_ratio=cfg.training.test_subject_ratio,
            seed=args.seed,
        )
        fold_results: list = []

        for fold, (train_idx, val_idx, test_idx) in enumerate(cv.split(X, y, subject_ids)):
            logger.info(f"\n=== FOLD {fold + 1}/{cfg.training.n_splits} ===")
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            model = None
            history = None
            feat_pipeline = None

            if not args.skip_train:
                if is_classical:
                    model, feat_pipeline = step_train_classical(
                        X_train, y_train, X_val, y_val,
                        cfg.model, cfg.features, cfg.data.target_sfreq,
                        output_dir / f"fold_{fold + 1}"
                    )
                else:
                    cfg.model.n_channels = X_train.shape[1]
                    cfg.model.n_timepoints = X_train.shape[2]
                    trainer, history = step_train_deep_learning(
                        X_train, y_train, X_val, y_val,
                        cfg.model, cfg.training,
                        output_dir / f"fold_{fold + 1}"
                    )
                    model, feat_pipeline = trainer, None

            # Baholash
            fold_metrics = step_evaluate(
                model, X_test, y_test,
                is_deep_learning=not is_classical,
                feat_pipeline=feat_pipeline if is_classical else None,
                dataset_name=dataset_info.name,
                model_name=args.model,
                history=history if not is_classical else None,
                output_dir=output_dir / f"fold_{fold + 1}",
            )
            fold_results.append({"fold": fold + 1, **fold_metrics})

        # Umumiy CV natija
        f1_scores = [f.get("macro_f1", 0.0) for f in fold_results]
        logger.info(f"\n=== CROSS-VALIDATION YAKUNIY NATIJA ===")
        logger.info(f"Macro F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    else:
        # Oddiy Train/Val/Test split (subject-wise)
        cv = SubjectWiseCrossValidator(n_splits=1, test_ratio=0.2, seed=args.seed)
        train_idx, val_idx, test_idx = next(cv.split(X, y, subject_ids))

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        history = None
        feat_pipeline = None

        if not args.skip_train:
            if is_classical:
                model, feat_pipeline = step_train_classical(
                    X_train, y_train, X_val, y_val,
                    cfg.model, cfg.features, cfg.data.target_sfreq, output_dir
                )
            else:
                cfg.model.n_channels = X_train.shape[1]
                cfg.model.n_timepoints = X_train.shape[2]
                trainer, history = step_train_deep_learning(
                    X_train, y_train, X_val, y_val,
                    cfg.model, cfg.training, output_dir
                )
                model = trainer

        # Baholash
        final_metrics = step_evaluate(
            model, X_test, y_test,
            is_deep_learning=not is_classical,
            feat_pipeline=feat_pipeline,
            dataset_name=dataset_info.name,
            model_name=args.model,
            history=history,
            output_dir=output_dir,
        )

        logger.info(
            f"\n=== YAKUNIY NATIJA ===\n"
            f"Accuracy: {final_metrics.get('accuracy', 0):.4f}\n"
            f"Macro F1: {final_metrics.get('macro_f1', 0):.4f}\n"
            f"AUROC:    {final_metrics.get('auroc', 0):.4f}"
        )

    logger.info("Pipeline muvaffaqiyatli yakunlandi.")


if __name__ == "__main__":
    main()
