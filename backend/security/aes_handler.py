"""
backend/security/aes_handler.py
=================================
AES-256-GCM (Galois/Counter Mode) orqali EEG fayllarini shifrlash va ochish.

Nima uchun AES-GCM?
  - Authenticated Encryption: Shifrlash + buzilishni aniqlash birgalikda.
  - 256-bit kalit → Harbiy sinfli himoya.
  - GCM → IV (nonce) har doim tasodifiy, takrorlanmas.
  - HIPAA va GDPR talablariga javob beradi.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# IV (nonce) uzunligi — GCM uchun standart 12 bayt
_GCM_NONCE_SIZE = 12
# Metadata ajratuvchi
_META_SEPARATOR = b"||META||"


def encrypt_bytes(data: bytes, key: bytes, metadata: dict | None = None) -> bytes:
    """
    Berilgan bytes ma'lumotni AES-256-GCM bilan shifrlaydi.

    Returns:
        nonce (12 bytes) + ciphertext + auth_tag — hammasi birga.
        Agar metadata bo'lsa, u biriktirilgan (associated data) sifatida ishlatiladi.
    """
    if len(key) != 32:
        raise ValueError(f"AES-256 uchun 32 baytlik kalit kerak, {len(key)} bayt berildi.")

    aesgcm = AESGCM(key)
    nonce = os.urandom(_GCM_NONCE_SIZE)

    # Metadata ni AAD (Additional Authenticated Data) sifatida qo'shamiz
    # Bu metadata shifrlash bilan bog'lanadi, lekin ochiq holatda saqlanadi
    aad = None
    if metadata:
        aad = json.dumps(metadata, sort_keys=True, ensure_ascii=True).encode("utf-8")

    ciphertext = aesgcm.encrypt(nonce, data, aad)

    # Paket: nonce + ciphertext (ichida auth tag ham bor)
    packet = nonce + ciphertext

    # Agar metadata bo'lsa, uni oxiriga qo'shamiz (unencrypted, lekin authenticated)
    if aad:
        packet += _META_SEPARATOR + aad

    return packet


def decrypt_bytes(packet: bytes, key: bytes) -> tuple[bytes, dict | None]:
    """
    AES-256-GCM bilan shifrlangan paketni ochadi.

    Returns:
        (plaintext_bytes, metadata_dict | None)

    Raises:
        cryptography.exceptions.InvalidTag — agar kalit noto'g'ri yoki ma'lumot buzilgan.
    """
    if len(key) != 32:
        raise ValueError(f"AES-256 uchun 32 baytlik kalit kerak, {len(key)} bayt berildi.")

    # Metadatani ajratib olamiz
    aad = None
    metadata = None
    if _META_SEPARATOR in packet:
        packet, aad = packet.split(_META_SEPARATOR, 1)
        try:
            metadata = json.loads(aad.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            metadata = None

    nonce = packet[:_GCM_NONCE_SIZE]
    ciphertext = packet[_GCM_NONCE_SIZE:]

    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, aad)

    return plaintext, metadata


def encrypt_file(
    source_path: str | Path,
    dest_dir: str | Path,
    key: bytes,
    patient_id: str = "unknown",
    delete_source: bool = False,
) -> Path:
    """
    EEG faylini (EDF/SET/BDF) AES-256-GCM bilan shifrlaydi va saqlaydi.

    Args:
        source_path: Shifrlanuvchi fayl yo'li.
        dest_dir: Shifrlanagan faylni saqlash papkasi.
        key: 32 baytlik AES-256 kalit.
        patient_id: Bemor identifikatori (metadata uchun).
        delete_source: True bo'lsa, asl faylni o'chiradi.

    Returns:
        Shifrlanagan fayl yo'li (.enc kengaytmali).
    """
    source_path = Path(source_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    raw_data = source_path.read_bytes()

    metadata = {
        "original_filename": source_path.name,
        "original_size_bytes": len(raw_data),
        "patient_id": patient_id,
        "encrypted_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "AES-256-GCM",
    }

    encrypted = encrypt_bytes(raw_data, key, metadata=metadata)

    dest_path = dest_dir / (source_path.name + ".enc")
    dest_path.write_bytes(encrypted)

    if delete_source:
        source_path.unlink()

    return dest_path


def decrypt_file(
    encrypted_path: str | Path,
    dest_dir: str | Path,
    key: bytes,
) -> Path:
    """
    Shifrlanagan EEG faylini ochadi va saqlaydi.

    Args:
        encrypted_path: .enc kengaytmali shifrlangan fayl.
        dest_dir: Ochilgan faylni saqlash papkasi.
        key: 32 baytlik AES-256 kalit.

    Returns:
        Ochilgan fayl yo'li.
    """
    encrypted_path = Path(encrypted_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    packet = encrypted_path.read_bytes()
    plaintext, metadata = decrypt_bytes(packet, key)

    # Asl fayl nomini metadata dan olamiz
    if metadata and "original_filename" in metadata:
        original_name = metadata["original_filename"]
    else:
        # .enc ni olib tashlaymiz
        original_name = encrypted_path.stem

    dest_path = dest_dir / original_name
    dest_path.write_bytes(plaintext)

    return dest_path


def encrypt_numpy_array(arr_bytes: bytes, key: bytes, context: str = "") -> str:
    """
    NumPy array bytes'ini shifrlaydi va Base64 string qaytaradi.
    HTTP API orqali EEG ma'lumotlarini xavfsiz yuborish uchun.
    """
    meta = {"context": context, "size": len(arr_bytes)}
    encrypted = encrypt_bytes(arr_bytes, key, metadata=meta)
    return base64.b64encode(encrypted).decode("ascii")


def decrypt_numpy_array(encrypted_b64: str, key: bytes) -> bytes:
    """Base64 shifrlangan string'ni ochib, numpy bytes qaytaradi."""
    packet = base64.b64decode(encrypted_b64.encode("ascii"))
    plaintext, _ = decrypt_bytes(packet, key)
    return plaintext
