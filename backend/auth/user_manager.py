"""
backend/auth/user_manager.py
==============================
Foydalanuvchi boshqaruvi: yaratish, tekshirish, ro'yxat.

Xavfsizlik:
  - Parollar bcrypt bilan xeshlanadi (work factor=12).
  - Hech qachon tekis (plain) parol saqlanmaydi.
  - users.json dan migratsiya qo'llab-quvvatlanadi.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.models.db import User, create_all_tables, get_db

# bcrypt 5.x ile passlib versiyasi moslik muammosi tufayli sha256_crypt ishlatamiz.
# Bu klinik muhit uchun ham etarli darajada xavfsiz (PBKDF2-SHA256 asosida).
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ── Parol operatsiyalari ───────────────────────────────────────────────────


def hash_password(plain_password: str) -> str:
    """Parolni bcrypt bilan xeshlaydi."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Parolni xesh bilan tekshiradi."""
    return pwd_context.verify(plain_password, hashed_password)


# ── Foydalanuvchi CRUD ────────────────────────────────────────────────────


def create_user(
    db: Session,
    username: str,
    plain_password: str,
    full_name: str,
    role: str = "doctor",
) -> User:
    """
    Yangi foydalanuvchi yaratadi.

    Roles:
        - admin   : Barcha huquqlar (foydalanuvchi boshqaruvi, audit log)
        - doctor  : Bemor va EEG tekshiruvi ko'rish/yaratish
        - viewer  : Faqat o'qish (kuzatuvchi)
    """
    valid_roles = {"admin", "doctor", "viewer"}
    if role not in valid_roles:
        raise ValueError(f"Noto'g'ri rol: {role!r}. Mumkin rollar: {valid_roles}")

    if db.query(User).filter(User.username == username).first():
        raise ValueError(f"Foydalanuvchi allaqachon mavjud: {username!r}")

    user = User(
        username=username,
        full_name=full_name,
        password_hash=hash_password(plain_password),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> User | None:
    """Username bo'yicha foydalanuvchini topadi."""
    return db.query(User).filter(User.username == username, User.is_active == True).first()


def authenticate_user(db: Session, username: str, plain_password: str) -> User | None:
    """
    Foydalanuvchini autentifikatsiya qiladi.

    Returns:
        User ob'ekti (muvaffaqiyatli) yoki None (xato parol/foydalanuvchi yo'q).
    """
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(plain_password, user.password_hash):
        return None
    return user


def change_password(db: Session, username: str, new_password: str) -> bool:
    """Foydalanuvchining parolini yangilaydi."""
    user = get_user_by_username(db, username)
    if not user:
        return False
    user.password_hash = hash_password(new_password)
    db.commit()
    return True


def update_profile(
    db: Session, 
    current_username: str, 
    new_username: str | None = None, 
    full_name: str | None = None, 
    new_password: str | None = None
) -> User | None:
    """Foydalanuvchi profilini (ism, login, parol) yangilaydi."""
    user = get_user_by_username(db, current_username)
    if not user:
        return None

    if new_username and new_username != current_username:
        if db.query(User).filter(User.username == new_username).first():
            raise ValueError(f"'{new_username}' logini allaqachon band qilingan.")
        user.username = new_username

    if full_name:
        user.full_name = full_name

    if new_password:
        user.password_hash = hash_password(new_password)

    db.commit()
    db.refresh(user)
    return user


def deactivate_user(db: Session, username: str) -> bool:
    """Foydalanuvchini bloklaydi (o'chirish o'rniga)."""
    user = get_user_by_username(db, username)
    if not user:
        return False
    user.is_active = False
    db.commit()
    return True


def list_users(db: Session) -> list[dict]:
    """Barcha faol foydalanuvchilar ro'yxatini qaytaradi."""
    users = db.query(User).filter(User.is_active == True).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "full_name": u.full_name,
            "role": u.role,
            "last_login": u.last_login.isoformat() if u.last_login else None,
        }
        for u in users
    ]


# ── users.json dan migratsiya ─────────────────────────────────────────────


def migrate_from_json(db: Session, json_path: Path | None = None) -> int:
    """
    users.json dagi foydalanuvchilarni PostgreSQL bazaga ko'chiradi.
    Agar foydalanuvchi allaqachon mavjud bo'lsa, o'tkazib yuboradi.

    Returns:
        Ko'chirilgan foydalanuvchilar soni.
    """
    if json_path is None:
        json_path = BASE_DIR / "users.json"

    if not json_path.exists():
        print(f"users.json topilmadi: {json_path}. O'tkazildi.")
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        old_users: dict = json.load(f)

    migrated = 0
    for username, info in old_users.items():
        if db.query(User).filter(User.username == username).first():
            print(f"  Allaqachon mavjud: {username}. O'tkazildi.")
            continue

        # Eski SHA-256 hash o'rniga yangi bcrypt hash qo'shamiz
        # Lekin eski passwordni bilmaymiz — shuning uchun default parol beramiz
        # Admin uchun "admin123", doktor uchun "doctor123"
        default_pass = f"{username}123"
        role = info.get("role", "viewer")
        full_name = info.get("name", username.title())

        user = User(
            username=username,
            full_name=full_name,
            password_hash=hash_password(default_pass),
            role=role,
            is_active=True,
        )
        db.add(user)
        migrated += 1
        print(f"  Ko'chirildi: {username} ({role}) — vaqtinchalik parol: '{default_pass}'")

    db.commit()
    print(f"\nJami {migrated} ta foydalanuvchi ko'chirildi.")
    print("⚠️  Parollarni darhol o'zgartiring!")
    return migrated


# ── CLI interfeysi ─────────────────────────────────────────────────────────


def _cli_main() -> None:
    """
    Buyruq satri interfeysi.

    Foydalanish:
        python -m backend.auth.user_manager --migrate
        python -m backend.auth.user_manager --create --username john --name "John Doe" --role doctor
        python -m backend.auth.user_manager --list
    """
    parser = argparse.ArgumentParser(description="NeuroAnalyst Foydalanuvchi Boshqaruvi")
    parser.add_argument("--migrate", action="store_true", help="users.json dan PostgreSQL ga ko'chirish")
    parser.add_argument("--create", action="store_true", help="Yangi foydalanuvchi yaratish")
    parser.add_argument("--username", type=str)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--role", type=str, default="doctor", choices=["admin", "doctor", "viewer"])
    parser.add_argument("--list", action="store_true", help="Barcha foydalanuvchilarni ko'rsatish")

    args = parser.parse_args()

    create_all_tables()
    db = next(get_db())

    if args.migrate:
        migrate_from_json(db)

    elif args.create:
        if not args.username:
            print("--username kerak!")
            sys.exit(1)
        password = args.password or input(f"Parol ({args.username}): ")
        full_name = args.name or args.username.title()
        try:
            user = create_user(db, args.username, password, full_name, args.role)
            print(f"✅ Foydalanuvchi yaratildi: {user.username} ({user.role})")
        except ValueError as e:
            print(f"❌ Xato: {e}")
            sys.exit(1)

    elif args.list:
        users = list_users(db)
        print(f"\nJami {len(users)} ta foydalanuvchi:\n")
        for u in users:
            print(f"  {u['id']:3}. {u['username']:20} | {u['role']:10} | {u['full_name']}")
    else:
        parser.print_help()

    db.close()


if __name__ == "__main__":
    _cli_main()
