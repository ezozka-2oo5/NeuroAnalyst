"""
backend/auth/jwt_handler.py
=============================
JWT (JSON Web Token) orqali sessiya boshqaruvi.

Arxitektura:
  - Access Token: Qisqa muddatli (30 daqiqa) — har so'rovda yuboriladi.
  - Refresh Token: Uzoq muddatli (24 soat) — faqat tokenni yangilash uchun.
  - Token Blacklist: Logout qilingan tokenlar ma'lumotlar bazasida saqlanadi.
  - JTI (JWT ID): Har bir tokenning noyob identifikatori.

Xavfsizlik:
  - HS256 algoritmi (HMAC-SHA256) — tez va klinik darajada ishonchli.
  - `exp`, `iat`, `jti`, `sub`, `role` claimlar — to'liq validatsiya.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.models.db import TokenBlacklist

settings = get_settings()


# ── Token yaratish ─────────────────────────────────────────────────────────


def _create_token(
    subject: str,
    role: str,
    expires_delta: timedelta,
    token_type: str = "access",
    extra_claims: dict | None = None,
) -> tuple[str, str]:
    """
    JWT token yaratadi.

    Returns:
        (encoded_token, jti) tuple
    """
    now = datetime.now(timezone.utc)
    jti = str(uuid.uuid4())

    payload: dict[str, Any] = {
        "sub": subject,          # subject (username)
        "role": role,            # foydalanuvchi roli
        "type": token_type,      # "access" yoki "refresh"
        "jti": jti,              # noyob token ID (blacklist uchun)
        "iat": now,              # yaratilish vaqti
        "exp": now + expires_delta,  # tugash vaqti
    }

    if extra_claims:
        payload.update(extra_claims)

    encoded = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
    return encoded, jti


def create_access_token(username: str, role: str) -> str:
    """Qisqa muddatli access token yaratadi."""
    token, _ = _create_token(
        subject=username,
        role=role,
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
        token_type="access",
    )
    return token


def create_refresh_token(username: str, role: str) -> tuple[str, str]:
    """
    Uzoq muddatli refresh token yaratadi.

    Returns:
        (encoded_token, jti) — jti ni blacklist uchun saqlash kerak.
    """
    return _create_token(
        subject=username,
        role=role,
        expires_delta=timedelta(hours=settings.refresh_token_expire_hours),
        token_type="refresh",
    )


def decode_token(token: str) -> dict[str, Any]:
    """
    JWT tokenni tekshirib, payload qaytaradi.

    Raises:
        JWTError — agar token noto'g'ri, muddati o'tgan yoki noto'g'ri algoritm.
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
            options={"require": ["exp", "iat", "sub", "jti", "role", "type"]},
        )
        return payload
    except JWTError as e:
        raise JWTError(f"Token validatsiyasi muvaffaqiyatsiz: {e}") from e


def verify_access_token(token: str, db: Session) -> dict[str, Any]:
    """
    Access token'ni tekshiradi:
    1. Muddati o'tmaganmi?
    2. Token blacklist'da emasmi?
    3. Token turi "access"mi?

    Returns:
        Payload dict (sub, role, jti, ...)

    Raises:
        JWTError — agar tekshiruv o'tmasa.
    """
    payload = decode_token(token)

    if payload.get("type") != "access":
        raise JWTError("Bu access token emas.")

    jti = payload.get("jti")
    if jti and _is_blacklisted(jti, db):
        raise JWTError("Token bekor qilingan (logout amalga oshirilgan).")

    return payload


def blacklist_token(jti: str, expires_at: datetime, db: Session) -> None:
    """Tokenni blacklist'ga qo'shadi (logout uchun)."""
    record = TokenBlacklist(jti=jti, expires_at=expires_at)
    db.add(record)
    db.commit()


def cleanup_expired_blacklist(db: Session) -> int:
    """Muddati o'tgan blacklist yozuvlarini o'chiradi. Qaytarilgan qiymat — o'chirilgan yozuvlar soni."""
    now = datetime.now(timezone.utc)
    deleted = db.query(TokenBlacklist).filter(TokenBlacklist.expires_at < now).delete()
    db.commit()
    return deleted


def _is_blacklisted(jti: str, db: Session) -> bool:
    """Token blacklist'da borligini tekshiradi."""
    return db.query(TokenBlacklist).filter(TokenBlacklist.jti == jti).first() is not None
