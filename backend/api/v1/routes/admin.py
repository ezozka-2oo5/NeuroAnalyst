"""
backend/api/v1/routes/admin.py
================================
Admin paneli endpointlari — faqat Admin roli uchun.

Endpointlar:
  GET    /api/v1/admin/users              → users:read     — Barcha foydalanuvchilar
  POST   /api/v1/admin/users              → users:write    — Yangi foydalanuvchi yaratish
  PUT    /api/v1/admin/users/{username}   → users:write    — Rolni/parolni o'zgartirish
  DELETE /api/v1/admin/users/{username}   → users:delete   — Bloklash (o'chirish emas)
  GET    /api/v1/admin/audit-logs         → audit:read     — Barcha harakatlar tarixi
  GET    /api/v1/admin/permissions        → (auth)         — Barcha rollar va ruxsatlar
  GET    /api/v1/admin/stats              → system:settings — Tizim statistikasi
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.auth.rbac import (
    ROLE_PERMISSIONS,
    Permission,
    get_user_permissions,
    require_permission,
)
from backend.auth.user_manager import change_password, create_user, deactivate_user, list_users
from backend.models.db import AuditLog, Patient, Study, User, get_db

router = APIRouter(prefix="/api/v1/admin", tags=["Admin Panel"])


# ── Pydantic schemalar ─────────────────────────────────────────────────────


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=128)
    role: str = Field(default="doctor", description="'admin', 'doctor' yoki 'viewer'")


class UserUpdateRequest(BaseModel):
    new_password: str | None = Field(default=None, min_length=8)
    new_role: str | None = Field(default=None)
    full_name: str | None = None


class UserResponse(BaseModel):
    id: int
    username: str
    full_name: str
    role: str
    is_active: bool
    last_login: str | None
    permissions: list[str]  # Rol asosida ruxsatlar ro'yxati


# ── Foydalanuvchi boshqaruvi ───────────────────────────────────────────────


@router.get(
    "/users",
    response_model=list[UserResponse],
    summary="Barcha foydalanuvchilar",
    description="Barcha faol foydalanuvchilar ro'yxati. **Ruxsat:** `users:read` (Admin only)",
)
def list_all_users(
    current_user: dict = Depends(require_permission(Permission.USERS_READ)),
    db: Session = Depends(get_db),
):
    users = db.query(User).all()
    return [
        UserResponse(
            id=u.id,
            username=u.username,
            full_name=u.full_name,
            role=u.role,
            is_active=u.is_active,
            last_login=u.last_login.isoformat() if u.last_login else None,
            permissions=get_user_permissions(u.role),
        )
        for u in users
    ]


@router.post(
    "/users",
    status_code=status.HTTP_201_CREATED,
    summary="Yangi foydalanuvchi yaratish",
    description="Yangi foydalanuvchi yaratadi. **Ruxsat:** `users:write` (Admin only)",
)
def create_new_user(
    data: UserCreateRequest,
    current_user: dict = Depends(require_permission(Permission.USERS_WRITE)),
    db: Session = Depends(get_db),
):
    valid_roles = {"admin", "doctor", "viewer"}
    if data.role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Noto'g'ri rol: '{data.role}'. Mumkin rollar: {sorted(valid_roles)}",
        )

    try:
        user = create_user(db, data.username, data.password, data.full_name, data.role)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    # Audit
    db.add(AuditLog(
        username=current_user["username"],
        action="create_user",
        detail=f"created='{data.username}' role='{data.role}'",
    ))
    db.commit()

    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "message": "Foydalanuvchi muvaffaqiyatli yaratildi.",
    }


@router.put(
    "/users/{username}",
    summary="Foydalanuvchi ma'lumotlarini yangilash",
    description="Foydalanuvchining rol, parol yoki ismini yangilaydi. **Ruxsat:** `users:write` (Admin only)",
)
def update_user(
    username: str,
    data: UserUpdateRequest,
    current_user: dict = Depends(require_permission(Permission.USERS_WRITE)),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Foydalanuvchi topilmadi.")

    changes = []

    if data.new_role:
        valid_roles = {"admin", "doctor", "viewer"}
        if data.new_role not in valid_roles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Noto'g'ri rol: '{data.new_role}'.",
            )
        old_role = user.role
        user.role = data.new_role
        changes.append(f"role: {old_role}→{data.new_role}")

    if data.full_name:
        user.full_name = data.full_name
        changes.append("full_name yangilandi")

    if data.new_password:
        from backend.auth.user_manager import hash_password
        user.password_hash = hash_password(data.new_password)
        changes.append("parol yangilandi")

    db.commit()

    # Audit
    db.add(AuditLog(
        username=current_user["username"],
        action="update_user",
        detail=f"target='{username}' changes=[{', '.join(changes)}]",
    ))
    db.commit()

    return {
        "username": user.username,
        "role": user.role,
        "permissions": get_user_permissions(user.role),
        "message": "Foydalanuvchi yangilandi.",
    }


@router.delete(
    "/users/{username}",
    summary="Foydalanuvchini bloklash",
    description="Foydalanuvchini deaktivatsiya qiladi (o'chirish emas — is_active=False). **Ruxsat:** `users:delete` (Admin only)",
)
def block_user(
    username: str,
    current_user: dict = Depends(require_permission(Permission.USERS_DELETE)),
    db: Session = Depends(get_db),
):
    # O'z-o'zini bloklashdan himoya
    if username == current_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O'z akkauntingizni bloklash mumkin emas.",
        )

    success = deactivate_user(db, username)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Foydalanuvchi topilmadi.")

    db.add(AuditLog(
        username=current_user["username"],
        action="block_user",
        detail=f"blocked='{username}'",
    ))
    db.commit()

    return {"message": f"'{username}' bloklandi. Endi tizimga kira olmaydi."}


# ── Audit Log ──────────────────────────────────────────────────────────────


@router.get(
    "/audit-logs",
    summary="Audit log",
    description="Barcha harakatlar tarixi. **Ruxsat:** `audit:read` (Admin only)",
)
def get_audit_logs(
    skip: int = 0,
    limit: int = 200,
    username_filter: str | None = None,
    action_filter: str | None = None,
    current_user: dict = Depends(require_permission(Permission.AUDIT_READ)),
    db: Session = Depends(get_db),
):
    query = db.query(AuditLog).order_by(AuditLog.created_at.desc())

    if username_filter:
        query = query.filter(AuditLog.username.ilike(f"%{username_filter}%"))
    if action_filter:
        query = query.filter(AuditLog.action.ilike(f"%{action_filter}%"))

    logs = query.offset(skip).limit(limit).all()
    return [
        {
            "id": log.id,
            "username": log.username,
            "action": log.action,
            "detail": log.detail,
            "ip_address": log.ip_address,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]


# ── Ruxsatlar ro'yxati ─────────────────────────────────────────────────────


@router.get(
    "/permissions",
    summary="Barcha rollar va ularning ruxsatlari",
    description="RBAC matritsasini ko'rsatadi. Autentifikatsiya talab qilinadi.",
)
def get_all_permissions(
    current_user: dict = Depends(require_permission(Permission.USERS_READ)),
):
    return {
        role: sorted([p.value for p in perms])
        for role, perms in ROLE_PERMISSIONS.items()
    }


# ── Tizim statistikasi ─────────────────────────────────────────────────────


@router.get(
    "/stats",
    summary="Tizim statistikasi",
    description="Foydalanuvchilar, bemorlar va tekshiruvlar soni. **Ruxsat:** `system:settings` (Admin only)",
)
def get_system_stats(
    current_user: dict = Depends(require_permission(Permission.SYSTEM_SETTINGS)),
    db: Session = Depends(get_db),
):
    user_count = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
    patient_count = db.query(func.count(Patient.id)).scalar()
    study_count = db.query(func.count(Study.id)).scalar()
    audit_count = db.query(func.count(AuditLog.id)).scalar()

    # Rollarga bo'lingan foydalanuvchilar soni
    role_breakdown = (
        db.query(User.role, func.count(User.id))
        .filter(User.is_active == True)
        .group_by(User.role)
        .all()
    )

    return {
        "total_users": user_count,
        "total_patients": patient_count,
        "total_studies": study_count,
        "total_audit_logs": audit_count,
        "users_by_role": {role: count for role, count in role_breakdown},
    }
