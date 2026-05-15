"""
backend/api/v1/routes/patients.py
=====================================
Bemor registri endpointlari — To'liq RBAC bilan.

Ruxsatlar:
  GET    /api/v1/patients/              → patients:read       (admin, doctor, viewer)
  GET    /api/v1/patients/{id}          → patients:read       (admin, doctor, viewer)
  POST   /api/v1/patients/              → patients:write      (admin, doctor)
  PUT    /api/v1/patients/{id}          → patients:write      (admin, doctor)
  DELETE /api/v1/patients/{id}          → patients:delete     (admin only)
  GET    /api/v1/patients/{id}/studies  → studies:read        (admin, doctor, viewer)
  DELETE /api/v1/patients/{id}/studies/{sid} → studies:write  (admin, doctor)
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.auth.rbac import Permission, require_permission
from backend.models.db import AuditLog, Patient, Study, get_db

router = APIRouter(prefix="/api/v1/patients", tags=["Bemorlar"])


# ── Pydantic schemalar ─────────────────────────────────────────────────────


class PatientCreate(BaseModel):
    patient_code: str
    full_name: str
    age: int | None = None
    gender: str | None = None
    department: str | None = None
    clinical_note: str | None = None


class PatientUpdate(BaseModel):
    full_name: str | None = None
    age: int | None = None
    gender: str | None = None
    department: str | None = None
    clinical_note: str | None = None


class StudyCreate(BaseModel):
    source_name: str | None = None
    source_type: str | None = None
    duration_sec: float | None = None
    sfreq: float | None = None
    channel_count: int | None = None
    signal_quality: float | None = None
    artifact_ratio: float | None = None
    seizure_risk: float | None = None
    label: str | None = None
    confidence: float | None = None
    dominant_channel: str | None = None
    recommendation: str | None = None
    report_text: str | None = None
    ml_seizure_prob: float | None = None
    ml_label: str | None = None


# ── Yordamchi ─────────────────────────────────────────────────────────────


def _log(db: Session, username: str, action: str, detail: str) -> None:
    db.add(AuditLog(username=username, action=action, detail=detail))
    db.commit()


# ── Endpointlar ────────────────────────────────────────────────────────────


@router.get(
    "/",
    summary="Barcha bemorlar ro'yxati",
    description="Bemor registridagi barcha bemorlarni qaytaradi. **Ruxsat:** `patients:read`",
)
def list_patients(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(require_permission(Permission.PATIENTS_READ)),
    db: Session = Depends(get_db),
):
    patients = db.query(Patient).order_by(Patient.updated_at.desc()).offset(skip).limit(limit).all()
    return [
        {
            "id": p.id,
            "patient_code": p.patient_code,
            "full_name": p.full_name,
            "age": p.age,
            "gender": p.gender,
            "department": p.department,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in patients
    ]


@router.get(
    "/{patient_id}",
    summary="Bitta bemor ma'lumoti",
    description="Bemor ID bo'yicha to'liq ma'lumot qaytaradi. **Ruxsat:** `patients:read`",
)
def get_patient(
    patient_id: int,
    current_user: dict = Depends(require_permission(Permission.PATIENTS_READ)),
    db: Session = Depends(get_db),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bemor topilmadi.")
    return {
        "id": patient.id,
        "patient_code": patient.patient_code,
        "full_name": patient.full_name,
        "age": patient.age,
        "gender": patient.gender,
        "department": patient.department,
        "clinical_note": patient.clinical_note,
        "created_at": patient.created_at.isoformat() if patient.created_at else None,
        "updated_at": patient.updated_at.isoformat() if patient.updated_at else None,
    }


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Yangi bemor qo'shish",
    description="Registrga yangi bemor qo'shadi. **Ruxsat:** `patients:write` (Doctor, Admin)",
)
def create_patient(
    data: PatientCreate,
    current_user: dict = Depends(require_permission(Permission.PATIENTS_WRITE)),
    db: Session = Depends(get_db),
):
    existing = db.query(Patient).filter(Patient.patient_code == data.patient_code).first()
    if existing:
        # Upsert — mavjud bo'lsa yangilaydi
        for field, value in data.model_dump(exclude_none=True).items():
            if field != "patient_code":
                setattr(existing, field, value)
        existing.updated_at = datetime.now(timezone.utc)
        db.commit()
        _log(db, current_user["username"], "update_patient", f"id={existing.id} code={data.patient_code}")
        return {"id": existing.id, "created": False, "message": "Bemor yangilandi."}

    patient = Patient(**data.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    _log(db, current_user["username"], "create_patient", f"id={patient.id} code={data.patient_code}")
    return {"id": patient.id, "created": True, "message": "Bemor ro'yxatga qo'shildi."}


@router.put(
    "/{patient_id}",
    summary="Bemor ma'lumotlarini yangilash",
    description="Bemor ma'lumotlarini yangilaydi. **Ruxsat:** `patients:write` (Doctor, Admin)",
)
def update_patient(
    patient_id: int,
    data: PatientUpdate,
    current_user: dict = Depends(require_permission(Permission.PATIENTS_WRITE)),
    db: Session = Depends(get_db),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bemor topilmadi.")

    for field, value in data.model_dump(exclude_none=True).items():
        setattr(patient, field, value)
    patient.updated_at = datetime.now(timezone.utc)
    db.commit()
    _log(db, current_user["username"], "update_patient", f"id={patient_id}")
    return {"message": "Bemor ma'lumotlari yangilandi."}


@router.delete(
    "/{patient_id}",
    summary="Bemorni o'chirish",
    description="Bemorni bazadan o'chiradi. **Ruxsat:** `patients:delete` (faqat Admin)",
)
def delete_patient(
    patient_id: int,
    current_user: dict = Depends(require_permission(Permission.PATIENTS_DELETE)),
    db: Session = Depends(get_db),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bemor topilmadi.")

    # Bog'liq tekshiruvlarni ham o'chiramiz
    db.query(Study).filter(Study.patient_id == patient_id).delete()
    db.delete(patient)
    db.commit()
    _log(db, current_user["username"], "delete_patient", f"id={patient_id}")
    return {"message": f"Bemor #{patient_id} o'chirildi."}


@router.get(
    "/{patient_id}/studies",
    summary="Bemor EEG tekshiruvlari tarixi",
    description="Bemor barcha EEG tekshiruvlarini qaytaradi. **Ruxsat:** `studies:read`",
)
def get_patient_studies(
    patient_id: int,
    current_user: dict = Depends(require_permission(Permission.STUDIES_READ)),
    db: Session = Depends(get_db),
):
    # Bemor mavjudligini tekshirish
    if not db.query(Patient).filter(Patient.id == patient_id).first():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bemor topilmadi.")

    studies = (
        db.query(Study)
        .filter(Study.patient_id == patient_id)
        .order_by(Study.created_at.desc())
        .all()
    )
    return [
        {
            "id": s.id,
            "source_name": s.source_name,
            "label": s.label,
            "ml_label": s.ml_label,
            "seizure_risk": s.seizure_risk,
            "ml_seizure_prob": s.ml_seizure_prob,
            "signal_quality": s.signal_quality,
            "confidence": s.confidence,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in studies
    ]


@router.post(
    "/{patient_id}/studies",
    status_code=status.HTTP_201_CREATED,
    summary="Yangi EEG tekshiruvini saqlash",
    description="Bemorga yangi EEG tekshiruv natijasini qo'shadi. **Ruxsat:** `studies:write` (Doctor, Admin)",
)
def create_study(
    patient_id: int,
    data: StudyCreate,
    current_user: dict = Depends(require_permission(Permission.STUDIES_WRITE)),
    db: Session = Depends(get_db),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bemor topilmadi.")

    study = Study(patient_id=patient_id, **data.model_dump())
    db.add(study)
    
    # Bemorning updated_at vaqtini yangilash
    patient.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(study)
    _log(db, current_user["username"], "create_study", f"study_id={study.id} patient_id={patient_id}")
    return {"id": study.id, "message": "EEG tekshiruvi saqlandi."}


@router.get(
    "/studies/recent",
    summary="So'nggi barcha tekshiruvlar ro'yxati",
    description="Tizimdagi so'nggi EEG tekshiruvlarini ko'rsatadi. **Ruxsat:** `studies:read`",
)
def get_recent_all_studies(
    limit: int = 50,
    current_user: dict = Depends(require_permission(Permission.STUDIES_READ)),
    db: Session = Depends(get_db),
):
    studies = (
        db.query(Study, Patient.full_name)
        .join(Patient)
        .order_by(Study.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": s.id,
            "patient_id": s.patient_id,
            "patient_name": full_name,
            "source_name": s.source_name,
            "label": s.label,
            "ml_label": s.ml_label,
            "seizure_risk": s.seizure_risk,
            "signal_quality": s.signal_quality,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s, full_name in studies
    ]


@router.delete(
    "/{patient_id}/studies/{study_id}",
    summary="EEG tekshiruvini o'chirish",
    description="Bemorning bitta tekshiruvini o'chiradi. **Ruxsat:** `studies:write` (Doctor, Admin)",
)
def delete_study(
    patient_id: int,
    study_id: int,
    current_user: dict = Depends(require_permission(Permission.STUDIES_WRITE)),
    db: Session = Depends(get_db),
):
    study = db.query(Study).filter(Study.id == study_id, Study.patient_id == patient_id).first()
    if not study:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tekshiruv topilmadi.")

    db.delete(study)
    db.commit()
    _log(db, current_user["username"], "delete_study", f"study_id={study_id} patient_id={patient_id}")
    return {"message": f"Tekshiruv #{study_id} o'chirildi."}
