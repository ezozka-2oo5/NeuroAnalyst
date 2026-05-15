"""
backend/api/v1/routes/auth.py
================================
Autentifikatsiya endpointlari:
  POST /auth/login     — JWT token olish
  POST /auth/refresh   — token yangilash
  POST /auth/logout    — token bekor qilish
  GET  /auth/me        — joriy foydalanuvchi ma'lumoti
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.auth.jwt_handler import (
    blacklist_token,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_access_token,
)
from backend.auth.user_manager import authenticate_user
from backend.models.db import AuditLog, User, get_db

router = APIRouter(prefix="/auth", tags=["Autentifikatsiya"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Pydantic schemalar ─────────────────────────────────────────────────────


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # soniyalarda
    user: dict


class RefreshRequest(BaseModel):
    refresh_token: str


class MeResponse(BaseModel):
    username: str
    full_name: str
    role: str
    last_login: str | None
    permissions: list[str]  # Rol asosida berilgan ruxsatlar


class ProfileUpdateRequest(BaseModel):
    username: str | None = None
    full_name: str | None = None
    password: str | None = None


# ── Dependency: joriy foydalanuvchini olish ────────────────────────────────


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> dict:
    """
    FastAPI dependency — JWT tokenni tekshirib, joriy foydalanuvchi ma'lumotini qaytaradi.

    Raises:
        HTTP 401 — agar token noto'g'ri yoki muddati o'tgan.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token yaroqsiz yoki muddati o'tgan.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = verify_access_token(token, db)
    except JWTError:
        raise credentials_exception

    username = payload.get("sub")
    role = payload.get("role")
    jti = payload.get("jti")

    if not username or not role:
        raise credentials_exception

    return {"username": username, "role": role, "jti": jti}


def require_role(*roles: str):
    """Faqat ko'rsatilgan rollar uchun ruxsat beruvchi dependency factory."""
    def _checker(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Bu amalni bajarish uchun yetarli huquq yo'q. Kerakli rol: {roles}",
            )
        return current_user
    return _checker


def _log_audit(db: Session, username: str, action: str, detail: str, ip: str = "") -> None:
    """Audit logga yozadi."""
    log = AuditLog(username=username, action=action, detail=detail, ip_address=ip)
    db.add(log)
    db.commit()


# ── Endpointlar ────────────────────────────────────────────────────────────


@router.post("/login", response_model=TokenResponse, summary="Kirish va JWT token olish")
def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Foydalanuvchi nomi va parol bilan kirish.

    - **username**: Foydalanuvchi nomi
    - **password**: Parol (bcrypt bilan tekshiriladi)

    Returns: access_token va refresh_token
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        ip = request.client.host if request.client else "unknown"
        _log_audit(db, form_data.username, "login_failed", f"ip={ip}", ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Foydalanuvchi nomi yoki parol noto'g'ri.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Access va refresh tokenlar
    access_token = create_access_token(user.username, user.role)
    refresh_token, _ = create_refresh_token(user.username, user.role)

    # Last login vaqtini yangilash
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    ip = request.client.host if request.client else "unknown"
    _log_audit(db, user.username, "login_success", f"role={user.role}, ip={ip}", ip)

    from backend.config import get_settings
    settings = get_settings()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60,
        user={"username": user.username, "full_name": user.full_name, "role": user.role},
    )


@router.post("/refresh", response_model=TokenResponse, summary="Access tokenni yangilash")
def refresh_token(
    body: RefreshRequest,
    db: Session = Depends(get_db),
):
    """
    Refresh token orqali yangi access token olish.
    Muddati o'tgan access token o'rniga foydalaniladi.
    """
    try:
        payload = decode_token(body.refresh_token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token yaroqsiz yoki muddati o'tgan.",
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bu refresh token emas.",
        )

    username = payload["sub"]
    role = payload["role"]

    new_access = create_access_token(username, role)
    new_refresh, _ = create_refresh_token(username, role)

    from backend.config import get_settings
    settings = get_settings()

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=settings.access_token_expire_minutes * 60,
        user={"username": username, "role": role},
    )


@router.post("/logout", summary="Chiqish (tokenni bekor qilish)")
def logout(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Joriy access tokenni blacklist'ga qo'shadi.
    Logout qilingan token qayta ishlatib bo'lmaydi.
    """
    jti = current_user.get("jti")
    if jti:
        from datetime import timedelta
        from backend.config import get_settings
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=get_settings().access_token_expire_minutes)
        blacklist_token(jti, expires_at, db)

    _log_audit(db, current_user["username"], "logout", "Token blacklist ga qo'shildi")
    return {"message": "Muvaffaqiyatli chiqildi."}


@router.get("/me", response_model=MeResponse, summary="Joriy foydalanuvchi ma'lumoti")
def get_me(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Kirgan foydalanuvchining profil ma'lumotlari va ruxsatlar ro'yxatini qaytaradi."""
    from backend.auth.rbac import get_user_permissions
    user = db.query(User).filter(User.username == current_user["username"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="Foydalanuvchi topilmadi.")
    return MeResponse(
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        last_login=user.last_login.isoformat() if user.last_login else None,
        permissions=get_user_permissions(user.role),
    )


@router.put("/me", summary="Joriy profilni tahrirlash")
def update_my_profile(
    body: ProfileUpdateRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Foydalanuvchining login, ism yoki parolini yangilash.
    """
    from backend.auth.user_manager import update_profile
    try:
        user = update_profile(
            db=db,
            current_username=current_user["username"],
            new_username=body.username,
            full_name=body.full_name,
            new_password=body.password,
        )
        if not user:
            raise HTTPException(status_code=404, detail="Foydalanuvchi topilmadi.")
        
        _log_audit(db, current_user["username"], "profile_update", f"Yangi login: {user.username}")
        return {"message": "Profil muvaffaqiyatli yangilandi.", "username": user.username}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/me/permissions", summary="Joriy foydalanuvchi ruxsatlari")
def get_my_permissions(
    current_user: dict = Depends(get_current_user),
):
    """
    Streamlit UI uchun qulay endpoint — kirgan foydalanuvchining
    barcha ruxsatlarini ro'yxat sifatida qaytaradi.

    Frontend bu ro'yxatga qarab qaysi UI elementlarini ko'rsatishni hal qiladi.
    """
    from backend.auth.rbac import get_user_permissions, has_permission, Permission
    role = current_user["role"]
    return {
        "role": role,
        "permissions": get_user_permissions(role),
        "can": {
            "patients_read":   has_permission(role, Permission.PATIENTS_READ),
            "patients_write":  has_permission(role, Permission.PATIENTS_WRITE),
            "patients_delete": has_permission(role, Permission.PATIENTS_DELETE),
            "studies_read":    has_permission(role, Permission.STUDIES_READ),
            "studies_write":   has_permission(role, Permission.STUDIES_WRITE),
            "inference_run":   has_permission(role, Permission.INFERENCE_RUN),
            "inference_status":has_permission(role, Permission.INFERENCE_STATUS),
            "reports_read":    has_permission(role, Permission.REPORTS_READ),
            "reports_generate":has_permission(role, Permission.REPORTS_GENERATE),
            "users_read":      has_permission(role, Permission.USERS_READ),
            "users_write":     has_permission(role, Permission.USERS_WRITE),
            "users_delete":    has_permission(role, Permission.USERS_DELETE),
            "audit_read":      has_permission(role, Permission.AUDIT_READ),
            "system_settings": has_permission(role, Permission.SYSTEM_SETTINGS),
        },
    }
