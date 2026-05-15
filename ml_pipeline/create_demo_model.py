"""
ml_pipeline/create_demo_model.py
================================
Serverda tez demo model yaratish uchun skript.
Bu skript minimal synthetic data bilan model o'qitadi va ONNX ga eksport qiladi.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

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

logger = get_logger(__name__)


def generate_minimal_synthetic_data(n_samples: int = 100, n_channels: int = 19, n_timepoints: int = 1024):
    """Minimal synthetic EEG data yaratadi."""
    logger.info(f"Generating {n_samples} synthetic samples...")
    X = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    y = np.random.randint(0, 2, size=n_samples)
    return X, y


def create_and_save_demo_model():
    """Demo model yaratadi va saqlaydi."""
    logger.info("=" * 60)
    logger.info("  Creating Demo Model for Server Deployment")
    logger.info("=" * 60)
    
    set_global_seed(42)
    
    # Konfiguratsiya
    cfg = PipelineConfig(
        data=DataConfig(target_sfreq=256.0, channels_to_use=None),
        preprocessing=PreprocessingConfig(
            apply_ica=False,
            epoch_duration_sec=4.0,
            epoch_overlap_sec=2.0,
        ),
        features=FeatureConfig(
            use_spectral=True,
            use_temporal=True,
            use_entropy=False,
            use_connectivity=False,
        ),
        model=ModelConfig(
            model_type="cnn_lstm",
            n_channels=19,
            n_timepoints=1024,
            cnn_filters=[32, 64],
            lstm_units=64,
            n_classes=2,
            final_dropout=0.4,
        ),
        training=TrainingConfig(
            epochs=5,  # Juda kam epoch - faqat model yaratish uchun
            batch_size=16,
            learning_rate=1e-3,
            early_stopping_patience=3,
            seed=42,
        ),
    )
    
    # Output directories
    demo_dir = BASE_DIR / "ml_pipeline" / "demo_output"
    checkpoint_dir = demo_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    cfg.save(demo_dir / "demo_config.json")
    logger.info(f"Config saved to: {demo_dir / 'demo_config.json'}")
    
    # Generate minimal data
    X_train, y_train = generate_minimal_synthetic_data(n_samples=80)
    X_val, y_val = generate_minimal_synthetic_data(n_samples=20)
    
    # Create and train model
    logger.info("Creating CNN-LSTM model...")
    from ml_pipeline.models.cnn_lstm import EEGCNNLSTM
    from ml_pipeline.training.trainer import EEGModelTrainer
    
    model = EEGCNNLSTM(cfg.model)
    trainer = EEGModelTrainer(
        cfg.training,
        model,
        checkpoint_dir=checkpoint_dir,
        log_dir=demo_dir / "logs",
    )
    
    logger.info("Training model (minimal epochs)...")
    t0 = time.time()
    history = trainer.fit(X_train, y_train, X_val, y_val)
    logger.info(f"Training completed in {time.time() - t0:.1f}s")
    
    # Verify checkpoint exists
    best_model_path = checkpoint_dir / "eeg_model_best.pt"
    if not best_model_path.exists():
        logger.error(f"Model checkpoint not found: {best_model_path}")
        return False
    
    logger.info(f"✅ Model checkpoint saved: {best_model_path}")
    
    # Export to ONNX
    logger.info("Exporting to ONNX...")
    try:
        from ml_pipeline.export_onnx import export_to_onnx
        
        onnx_path = export_to_onnx(
            pt_path=best_model_path,
            config_path=demo_dir / "demo_config.json",
            onnx_path=checkpoint_dir / "eeg_model.onnx",
            verify=False,  # Skip verification for speed
        )
        logger.info(f"✅ ONNX model saved: {onnx_path}")
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
        logger.warning("Continuing without ONNX export...")
    
    logger.info("\n" + "=" * 60)
    logger.info("  ✅ Demo Model Created Successfully!")
    logger.info("=" * 60)
    logger.info(f"  Model: {best_model_path}")
    logger.info(f"  Config: {demo_dir / 'demo_config.json'}")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = create_and_save_demo_model()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Failed to create demo model: {e}", exc_info=True)
        sys.exit(1)

# Made with Bob
