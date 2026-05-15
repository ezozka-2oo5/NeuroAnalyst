"""
backend/inference/onnx_runner.py
==================================
ONNX Runtime orqali EEG model inference.

Arxitektura:
  - Singleton pattern — model bir marta yuklanadi (xotira tejash).
  - Thread-safe — parallel so'rovlarda xavfsiz.
  - Auto-fallback: ONNX → PyTorch (agar ONNX fayl bo'lmasa).
  - Preprocessing — raw EEG → ONNX input transformation.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import get_settings

settings = get_settings()

_lock = threading.Lock()
_session = None          # onnxruntime InferenceSession
_pytorch_model = None    # Fallback PyTorch modeli
_device = None           # PyTorch device
_prep_pipeline = None    # PreprocessingPipeline
_n_classes = 3           # Sinf soni (default)
_model_type = "none"     # "onnx" | "pytorch" | "none"


def _load_onnx_session(onnx_path: str) -> object:
    """ONNX Runtime sessiyasini yaratadi."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2

    # CUDA mavjud bo'lsa GPU, aks holda CPU
    providers = []
    if ort.get_device() == "GPU":
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    return ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)


def _load_pytorch_fallback(pt_path: str, config_path: str):
    """PyTorch modelni fallback uchun yuklaydi."""
    import torch
    from ml_pipeline.models.cnn_lstm import EEGCNNLSTM
    from ml_pipeline.utils.config import PipelineConfig

    cfg = PipelineConfig.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGCNNLSTM(cfg.model)

    checkpoint = torch.load(pt_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    return model, device, cfg


def initialize_model() -> str:
    """
    Modelni birinchi marta yuklab, global singleton ga saqlaydi.

    Returns:
        "onnx" | "pytorch" | "none" — qaysi model yuklandi.
    """
    global _session, _pytorch_model, _device, _prep_pipeline, _n_classes, _model_type

    with _lock:
        if _model_type != "none":
            return _model_type

        onnx_path = settings.effective_onnx_path
        pt_path = settings.effective_model_path
        config_path = settings.effective_config_path

        # Preprocessing pipeline ni yuklash
        try:
            from ml_pipeline.utils.config import PipelineConfig
            from ml_pipeline.preprocessing.pipeline import PreprocessingPipeline
            cfg = PipelineConfig.load(config_path)
            _prep_pipeline = PreprocessingPipeline(cfg.data, cfg.preprocessing)
            _n_classes = cfg.model.n_classes
        except Exception as e:
            print(f"[ONNXRunner] Preprocessing pipeline yuklanmadi: {e}")

        # 1. ONNX ni sinab ko'ramiz
        if Path(onnx_path).exists():
            try:
                _session = _load_onnx_session(onnx_path)
                _model_type = "onnx"
                print(f"[ONNXRunner] ✅ ONNX model yuklandi: {onnx_path}")
                return "onnx"
            except Exception as e:
                print(f"[ONNXRunner] ONNX yuklanmadi: {e}. PyTorch fallback...")

        # 2. PyTorch fallback
        if Path(pt_path).exists() and Path(config_path).exists():
            try:
                _pytorch_model, _device, _ = _load_pytorch_fallback(pt_path, config_path)
                _model_type = "pytorch"
                print(f"[ONNXRunner] ✅ PyTorch model yuklandi (fallback): {pt_path}")
                return "pytorch"
            except Exception as e:
                print(f"[ONNXRunner] PyTorch ham yuklanmadi: {e}")

        _model_type = "none"
        print("[ONNXRunner] ⚠️  Model yuklanmadi — inference mavjud emas.")
        return "none"


def run_inference(eeg_array: np.ndarray) -> dict:
    """
    Raw EEG ma'lumotlarida bashorat qiladi.

    Args:
        eeg_array: (n_channels, n_timepoints) float32 — bir recording.

    Returns:
        {
            "probabilities": [p0, p1, p2, ...],  # Har sinf uchun ehtimollik
            "predicted_class": int,
            "model_type": "onnx" | "pytorch" | "none",
            "n_epochs": int,
        }
    """
    model_type = initialize_model()

    if model_type == "none":
        return {
            "probabilities": [1.0 / _n_classes] * _n_classes,
            "predicted_class": 0,
            "model_type": "none",
            "n_epochs": 0,
        }

    # EEG ni preprocessing orqali o'tkazamiz
    epochs = _preprocess_eeg(eeg_array)

    if epochs is None or len(epochs) == 0:
        return {
            "probabilities": [1.0 / _n_classes] * _n_classes,
            "predicted_class": 0,
            "model_type": model_type,
            "n_epochs": 0,
        }

    # Inference
    if model_type == "onnx":
        probs = _onnx_inference(epochs)
    else:
        probs = _pytorch_inference(epochs)

    # O'rtachani olamiz (barcha epochlar bo'yicha)
    mean_probs = np.mean(probs, axis=0).tolist()
    predicted_class = int(np.argmax(mean_probs))

    return {
        "probabilities": mean_probs,
        "predicted_class": predicted_class,
        "model_type": model_type,
        "n_epochs": len(epochs),
    }


def _preprocess_eeg(eeg_array: np.ndarray) -> Optional[np.ndarray]:
    """EEG ni preprocessing pipeline orqali o'tkazadi."""
    if _prep_pipeline is None:
        # Oddiy fallback — kanal/vaqt o'lchamini to'g'irlaymiz
        return _simple_preprocess(eeg_array)

    try:
        recording = {
            "data": eeg_array,
            "sfreq": 256.0,
            "channel_names": [f"CH{i}" for i in range(eeg_array.shape[0])],
            "source_name": "api_input",
            "label": 0,
        }
        X, _ = _prep_pipeline.process_dataset([recording], fit_normalizer=False)
        return X.astype(np.float32)
    except Exception as e:
        print(f"[ONNXRunner] Preprocessing xatosi: {e}")
        return _simple_preprocess(eeg_array)


def _simple_preprocess(eeg_array: np.ndarray) -> np.ndarray:
    """Oddiy preprocessing — model input o'lchamiga keltirish."""
    # (n_channels, n_timepoints) → (1, n_channels, n_timepoints)
    arr = eeg_array.astype(np.float32)
    # Z-score normalizatsiya
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + 1e-8
    arr = (arr - mean) / std
    return arr[np.newaxis, ...]  # (1, C, T)


def _onnx_inference(epochs: np.ndarray) -> np.ndarray:
    """ONNX Runtime orqali inference."""
    ort_inputs = {"eeg_input": epochs}
    logits = _session.run(["logits"], ort_inputs)[0]
    # Softmax
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probs


def _pytorch_inference(epochs: np.ndarray) -> np.ndarray:
    """PyTorch orqali inference (fallback)."""
    import torch
    x = torch.tensor(epochs, dtype=torch.float32).to(_device)
    with torch.no_grad():
        logits = _pytorch_model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def get_model_status() -> dict:
    """Model holati haqida ma'lumot qaytaradi."""
    return {
        "model_type": _model_type,
        "is_ready": _model_type != "none",
        "onnx_path": settings.effective_onnx_path,
        "pt_path": settings.effective_model_path,
        "onnx_exists": Path(settings.effective_onnx_path).exists(),
        "pt_exists": Path(settings.effective_model_path).exists(),
        "n_classes": _n_classes,
    }
