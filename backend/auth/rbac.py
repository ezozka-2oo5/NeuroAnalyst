"""
backend/auth/rbac.py
======================
To'liq RBAC (Role-Based Access Control) tizimi.

Arxitektura:
  - Permission  : Barcha mumkin ruxsatlar (Enum)
  - ROLE_PERMISSIONS : Har bir rol uchun ruxsatlar to'plami
  - require_permission() : FastAPI dependency factory
  - has_permission()     : Dastur ichida tekshirish uchun yordamchi

Ruxsat matritsasi:
┌─────────────────────────────────────────┬───────┬────────┬────────┐
│ Ruxsat                                  │ Admin │ Doctor │ Viewer │
├─────────────────────────────────────────┼───────┼────────┼────────┤
│ patients:read                           │  ✅   │   ✅   │   ✅   │
│ patients:write                          │  ✅   │   ✅   │   ❌   │
│ patients:delete                         │  ✅   │   ❌   │   ❌   │
│ studies:read                            │  ✅   │   ✅   │   ✅   │
│ studies:write                           │  ✅   │   ✅   │   ❌   │
│ inference:run                           │  ✅   │   ✅   │   ❌   │
│ inference:status                        │  ✅   │   ✅   │   ✅   │
│ reports:read                            │  ✅   │   ✅   │   ✅   │
│ reports:generate                        │  ✅   │   ✅   │   ❌   │
│ users:read                              │  ✅   │   ❌   │   ❌   │
│ users:write                             │  ✅   │   ❌   │   ❌   │
│ users:delete                            │  ✅   │   ❌   │   ❌   │
│ audit:read                              │  ✅   │   ❌   │   ❌   │
│ system:settings                         │  ✅   │   ❌   │   ❌   │
└─────────────────────────────────────────┴───────┴────────┴────────┘
"""

from __future__ import annotations

from enum import Enum

from fastapi import Depends, HTTPException, status

from backend.api.v1.routes.auth import get_current_user


# ── Ruxsatlar (Permissions) ────────────────────────────────────────────────

class Permission(str, Enum):
    """Tizimda mavjud barcha ruxsatlar."""

    # Bemorlar
    PATIENTS_READ   = "patients:read"
    PATIENTS_WRITE  = "patients:write"
    PATIENTS_DELETE = "patients:delete"

    # Tekshiruvlar (EEG Studies)
    STUDIES_READ  = "studies:read"
    STUDIES_WRITE = "studies:write"

    # ML Inference
    INFERENCE_RUN    = "inference:run"
    INFERENCE_STATUS = "inference:status"

    # Hisobotlar
    REPORTS_READ     = "reports:read"
    REPORTS_GENERATE = "reports:generate"

    # Foydalanuvchi boshqaruvi
    USERS_READ   = "users:read"
    USERS_WRITE  = "users:write"
    USERS_DELETE = "users:delete"

    # Tizim
    AUDIT_READ       = "audit:read"
    SYSTEM_SETTINGS  = "system:settings"


# ── Rol → Ruxsatlar matritsasi ─────────────────────────────────────────────

ROLE_PERMISSIONS: dict[str, frozenset[Permission]] = {
    "admin": frozenset({
        # Bemorlar — to'liq
        Permission.PATIENTS_READ,
        Permission.PATIENTS_WRITE,
        Permission.PATIENTS_DELETE,
        # Tekshiruvlar — to'liq
        Permission.STUDIES_READ,
        Permission.STUDIES_WRITE,
        # Inference — to'liq
        Permission.INFERENCE_RUN,
        Permission.INFERENCE_STATUS,
        # Hisobotlar — to'liq
        Permission.REPORTS_READ,
        Permission.REPORTS_GENERATE,
        # Foydalanuvchilar — to'liq
        Permission.USERS_READ,
        Permission.USERS_WRITE,
        Permission.USERS_DELETE,
        # Tizim — to'liq
        Permission.AUDIT_READ,
        Permission.SYSTEM_SETTINGS,
    }),

    "doctor": frozenset({
        # Bemorlar — o'qish va yozish (o'chirish yo'q)
        Permission.PATIENTS_READ,
        Permission.PATIENTS_WRITE,
        # Tekshiruvlar — o'qish va yozish
        Permission.STUDIES_READ,
        Permission.STUDIES_WRITE,
        # Inference — ishlatish
        Permission.INFERENCE_RUN,
        Permission.INFERENCE_STATUS,
        # Hisobotlar — o'qish va yaratish
        Permission.REPORTS_READ,
        Permission.REPORTS_GENERATE,
    }),

    "viewer": frozenset({
        # Bemorlar — faqat o'qish
        Permission.PATIENTS_READ,
        # Tekshiruvlar — faqat o'qish
        Permission.STUDIES_READ,
        # Inference — faqat holat ko'rish
        Permission.INFERENCE_STATUS,
        # Hisobotlar — faqat o'qish
        Permission.REPORTS_READ,
    }),
}


# ── Yordamchi funksiyalar ─────────────────────────────────────────────────


def has_permission(role: str, permission: Permission) -> bool:
    """
    Berilgan rol uchun ruxsat borligini tekshiradi.

    Args:
        role: Foydalanuvchi roli ('admin', 'doctor', 'viewer').
        permission: Tekshiriladigan ruxsat.

    Returns:
        True — ruxsat mavjud, False — yo'q yoki noma'lum rol.
    """
    role_perms = ROLE_PERMISSIONS.get(role, frozenset())
    return permission in role_perms


def get_user_permissions(role: str) -> list[str]:
    """
    Berilgan rol uchun barcha ruxsatlar ro'yxatini qaytaradi.
    Streamlit UI da nima ko'rsatish kerakligini aniqlash uchun ishlatiladi.
    """
    return sorted([p.value for p in ROLE_PERMISSIONS.get(role, frozenset())])


# ── FastAPI Dependency Factory ─────────────────────────────────────────────


def require_permission(*permissions: Permission):
    """
    FastAPI dependency factory — berilgan ruxsatlardan KAMIDA BITTASI bo'lsa o'tkazadi.

    Foydalanish:
        @router.get("/endpoint")
        def my_view(user = Depends(require_permission(Permission.PATIENTS_READ))):
            ...

        # Bir necha ruxsatdan biri kifoya:
        @router.delete("/endpoint")
        def delete_view(user = Depends(require_permission(Permission.PATIENTS_DELETE, Permission.SYSTEM_SETTINGS))):
            ...

    Raises:
        HTTP 401 — Autentifikatsiya qilinmagan.
        HTTP 403 — Ruxsat yo'q.
    """
    def _checker(current_user: dict = Depends(get_current_user)) -> dict:
        role = current_user.get("role", "")

        # Kamida bitta ruxsat tekshiriladi (OR logic)
        if not any(has_permission(role, p) for p in permissions):
            missing = " | ".join(p.value for p in permissions)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Ruxsat rad etildi.",
                    "required_permissions": [p.value for p in permissions],
                    "your_role": role,
                    "message": f"Ushbu amalni bajarish uchun '{missing}' ruxsati kerak.",
                },
            )
        return current_user

    return _checker


def require_all_permissions(*permissions: Permission):
    """
    Barcha ko'rsatilgan ruxsatlar bo'lishi shart bo'lganda (AND logic).

    Foydalanish:
        @router.post("/endpoint")
        def my_view(user = Depends(require_all_permissions(Permission.PATIENTS_WRITE, Permission.REPORTS_GENERATE))):
            ...
    """
    def _checker(current_user: dict = Depends(get_current_user)) -> dict:
        role = current_user.get("role", "")
        missing_perms = [p for p in permissions if not has_permission(role, p)]

        if missing_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Ruxsat rad etildi.",
                    "missing_permissions": [p.value for p in missing_perms],
                    "your_role": role,
                    "message": "Bu amal uchun barcha ruxsatlar talab etiladi.",
                },
            )
        return current_user

    return _checker
