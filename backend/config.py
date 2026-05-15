"""
backend/config.py
==================
Barcha muhit o'zgaruvchilari va konfiguratsiya sozlamalarini
pydantic-settings orqali boshqaradi.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Xavfsizlik ──────────────────────────────────────────────────
    secret_key: str = "change-me-to-a-very-long-random-secret-key-at-least-32-chars"
    aes_master_key: str = "change-me-to-64-hex-chars-aes256-master-encryption-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_hours: int = 24

    # ── Ma'lumotlar bazasi ──────────────────────────────────────────
    db_connection: str = "sqlite"
    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_database: str = "neuroanalyst_db"
    db_username: str = "postgres"
    db_password: str = "postgres"

    @property
    def database_url(self) -> str:
        if self.db_connection in ["pgsql", "postgresql", "postgres"]:
            return f"postgresql://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        return f"sqlite:///{BASE_DIR}/clinical_platform.db"

    # ── Model yo'llari ──────────────────────────────────────────────
    model_path: str = str(BASE_DIR / "ml_pipeline" / "real_output" / "checkpoints" / "eeg_model_best.pt")
    onnx_path: str = str(BASE_DIR / "ml_pipeline" / "real_output" / "checkpoints" / "eeg_model.onnx")
    model_config_path: str = str(BASE_DIR / "ml_pipeline" / "real_output" / "real_config.json")

    # Fallback — agar real model bo'lmasa, demo modelni ishlatish
    @property
    def effective_model_path(self) -> str:
        if Path(self.model_path).exists():
            return self.model_path
        demo = str(BASE_DIR / "ml_pipeline" / "demo_output" / "checkpoints" / "eeg_model_best.pt")
        return demo

    @property
    def effective_onnx_path(self) -> str:
        if Path(self.onnx_path).exists():
            return self.onnx_path
        demo = str(BASE_DIR / "ml_pipeline" / "demo_output" / "checkpoints" / "eeg_model.onnx")
        return demo

    @property
    def effective_config_path(self) -> str:
        if Path(self.model_config_path).exists():
            return self.model_config_path
        demo = str(BASE_DIR / "ml_pipeline" / "demo_output" / "demo_config.json")
        return demo

    # ── Server ──────────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    allowed_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    backend_url: str = "http://localhost:8000"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    # ── Shifrlash ────────────────────────────────────────────────────
    encrypt_eeg_files: bool = True
    encrypted_data_dir: str = str(BASE_DIR / "data" / "encrypted_eeg")

    @property
    def aes_key_bytes(self) -> bytes:
        """AES key'ni hex stringdan bytes'ga o'tkazadi."""
        key_hex = self.aes_master_key.replace("-", "")
        try:
            key = bytes.fromhex(key_hex)
        except ValueError:
            # Agar hex emas bo'lsa, UTF-8 sifatida olib, 32 baytga to'ldiramiz
            raw = self.aes_master_key.encode("utf-8")
            key = raw[:32].ljust(32, b"\x00")
        return key[:32]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
