"""
backend/api/v1/routes/inference.py
=====================================
EEG inference endpointlari — To'liq RBAC bilan.

Ruxsatlar:
  GET  /api/v1/inference/status   → inference:status  (admin, doctor, viewer)
  POST /api/v1/inference/analyze  → inference:run      (admin, doctor)
"""

from __future__ import annotations

import base64

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.rbac import Permission, require_permission
from backend.inference.onnx_runner import get_model_status, initialize_model, run_inference
from backend.models.db import AuditLog, get_db

router = APIRouter(prefix="/api/v1/inference", tags=["EEG Inference"])

# Sinf nomlari (ds004504 — Alzheimer/FTD dataset)
CLASS_NAMES = {
    0: "Alzheimer (A)",
    1: "Frontotemporal Dementia (F)",
    2: "Sog'lom (Control)",
}

SEIZURE_CLASS_NAMES = {
    0: "Normal EEG",
    1: "Epileptiform / Patologik",
}


# ── Pydantic schemalar ─────────────────────────────────────────────────────


class InferenceRequest(BaseModel):
    eeg_data_b64: str = Field(..., description="Base64 encoded numpy float32 array")
    n_channels: int = Field(..., gt=0, le=256, description="Kanal soni")
    n_timepoints: int = Field(..., gt=0, description="Vaqt nuqtalari soni")
    sfreq: float = Field(default=256.0, description="Sampling rate (Hz)")
    channel_names: list[str] | None = Field(default=None)
    patient_id: str = Field(default="unknown")


class InferenceResponse(BaseModel):
    probabilities: list[float]
    predicted_class: int
    predicted_label: str
    seizure_probability: float
    model_type: str
    n_epochs: int
    confidence: float


class ModelStatusResponse(BaseModel):
    is_ready: bool
    model_type: str
    onnx_exists: bool
    pt_exists: bool
    n_classes: int


# ── Endpointlar ────────────────────────────────────────────────────────────


@router.get(
    "/status",
    response_model=ModelStatusResponse,
    summary="Model holati",
    description="Model yuklanganmi va qaysi turdagi ekanligini ko'rsatadi. **Ruxsat:** `inference:status` (barcha rollar)",
)
def model_status(
    current_user: dict = Depends(require_permission(Permission.INFERENCE_STATUS)),
):
    return get_model_status()


@router.post(
    "/analyze",
    response_model=InferenceResponse,
    summary="EEG tahlil qilish",
    description="EEG signalini ONNX/PyTorch modeli orqali tahlil qiladi. **Ruxsat:** `inference:run` (Doctor, Admin)",
)
def analyze_eeg(
    request: InferenceRequest,
    current_user: dict = Depends(require_permission(Permission.INFERENCE_RUN)),
    db: Session = Depends(get_db),
):
    """
    EEG ma'lumotini tahlil qilib, klinik bashorat qaytaradi.

    **Request body:**
    - `eeg_data_b64`: Base64 encoded numpy float32 (n_channels × n_timepoints)
    - `n_channels`: Kanal soni (1-256)
    - `n_timepoints`: Vaqt nuqtalari soni

    **Returns:** Ehtimolliklar, bashorat sinfi, model turi va ishonch darajasi.
    """
    # Base64 → numpy
    try:
        raw_bytes = base64.b64decode(request.eeg_data_b64)
        eeg_array = np.frombuffer(raw_bytes, dtype=np.float32).reshape(
            request.n_channels, request.n_timepoints
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"EEG ma'lumotini decode qilishda xato: {e}",
        )

    # Inference
    try:
        result = run_inference(eeg_array)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference xatosi: {e}",
        )

    probs = result["probabilities"]
    pred_class = result["predicted_class"]
    n_classes = len(probs)

    # Label va seizure probability
    if n_classes == 2:
        label_map = SEIZURE_CLASS_NAMES
        seizure_prob = float(probs[1]) * 100.0 if len(probs) > 1 else 0.0
    elif n_classes == 3:
        label_map = CLASS_NAMES
        seizure_prob = float(probs[0] + probs[1]) * 100.0
    else:
        label_map = {i: f"Sinf-{i}" for i in range(n_classes)}
        seizure_prob = (1.0 - float(probs[0])) * 100.0 if probs else 0.0

    predicted_label = label_map.get(pred_class, f"Sinf-{pred_class}")
    confidence = float(max(probs)) * 100.0

    # Audit log
    db.add(AuditLog(
        username=current_user["username"],
        action="eeg_inference",
        detail=(
            f"patient_id={request.patient_id} | model={result['model_type']} "
            f"| pred={predicted_label} | conf={confidence:.1f}%"
        ),
    ))
    db.commit()

    return InferenceResponse(
        probabilities=probs,
        predicted_class=pred_class,
        predicted_label=predicted_label,
        seizure_probability=round(seizure_prob, 2),
        model_type=result["model_type"],
        n_epochs=result["n_epochs"],
        confidence=round(confidence, 2),
    )
