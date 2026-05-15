"""
backend/models/db.py
======================
SQLAlchemy ORM modellari va PostgreSQL ma'lumotlar bazasi init.

Jadvallar:
  - users        — foydalanuvchilar (bcrypt xeshlangan parol, rollar)
  - patients     — bemorlar registri
  - studies      — EEG tekshiruvlari
  - audit_logs   — barcha harakatlar tarixi
  - token_blacklist — logout qilingan tokenlar
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from backend.config import get_settings

settings = get_settings()

# ── Engine yaratish ────────────────────────────────────────────────────────

_is_sqlite = settings.database_url.startswith("sqlite")

_engine_kwargs = {
    "echo": False,
}

if _is_sqlite:
    # SQLite uchun maxsus sozlamalar (fallback rejimda)
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # PostgreSQL uchun connection pool sozlamalari
    _engine_kwargs["pool_size"] = 10
    _engine_kwargs["max_overflow"] = 20
    _engine_kwargs["pool_pre_ping"] = True
    _engine_kwargs["pool_recycle"] = 3600

engine = create_engine(settings.database_url, **_engine_kwargs)


# SQLite fallback uchun PRAGMA (faqat SQLite bilan ishlaganda ishlaydi)
if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ── Foydalanuvchilar ────────────────────────────────────────────────────────


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    full_name = Column(String(128), nullable=False)
    # bcrypt hash — hech qachon tekis parol saqlanmaydi
    password_hash = Column(String(256), nullable=False)
    role = Column(String(32), nullable=False, default="viewer")
    # Mumkin rollar: admin, doctor, viewer
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Munosabatlar
    audit_logs = relationship("AuditLog", back_populates="user", lazy="dynamic")


# ── Token blacklist ────────────────────────────────────────────────────────


class TokenBlacklist(Base):
    """Logout qilingan (bekor qilingan) JWT tokenlar ro'yxati."""

    __tablename__ = "token_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(128), unique=True, nullable=False, index=True)  # JWT ID
    expires_at = Column(DateTime(timezone=True), nullable=False)
    blacklisted_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ── Bemorlar ────────────────────────────────────────────────────────────────


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(64), unique=True, nullable=False, index=True)
    full_name = Column(String(256), nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String(32), nullable=True)
    department = Column(String(128), nullable=True)
    clinical_note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    studies = relationship("Study", back_populates="patient", lazy="dynamic")


# ── Tekshiruvlar (EEG studies) ─────────────────────────────────────────────


class Study(Base):
    __tablename__ = "studies"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    source_name = Column(String(256), nullable=True)
    source_type = Column(String(32), nullable=True)
    duration_sec = Column(Float, nullable=True)
    sfreq = Column(Float, nullable=True)
    channel_count = Column(Integer, nullable=True)
    signal_quality = Column(Float, nullable=True)
    artifact_ratio = Column(Float, nullable=True)
    seizure_risk = Column(Float, nullable=True)
    label = Column(String(128), nullable=True)
    confidence = Column(Float, nullable=True)
    dominant_channel = Column(String(64), nullable=True)
    recommendation = Column(Text, nullable=True)
    report_text = Column(Text, nullable=True)
    # ML natijalari
    ml_seizure_prob = Column(Float, nullable=True)
    ml_label = Column(String(128), nullable=True)
    # Shifrlanagan fayl yo'li (agar fayl saqlangan bo'lsa)
    encrypted_file_path = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    patient = relationship("Patient", back_populates="studies")


# ── Audit log ──────────────────────────────────────────────────────────────


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    username = Column(String(64), nullable=True)
    action = Column(String(128), nullable=False)
    detail = Column(Text, nullable=True)
    ip_address = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="audit_logs")


# ── DB init va dependency ──────────────────────────────────────────────────


def create_all_tables() -> None:
    """Barcha jadvallarni yaratadi (agar mavjud bo'lmasa)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — har so'rovda DB sessiyasi yaratadi va yopadi."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
