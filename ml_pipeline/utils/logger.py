"""
ml_pipeline/utils/logger.py
============================
Markazlashtirilgan logging konfiguratsiyasi.
Har bir modul shu funksiyadan logger oladi.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Formatlangan logger qaytaradi.

    Args:
        name: Logger nomi (odatda __name__ ishlatiladi).
        log_file: Agar berilsa, log faylga ham yoziladi.
        level: Logging darajasi (DEBUG, INFO, WARNING, ERROR).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Agar allaqachon handler bo'lsa, qayta qo'shmaymiz
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (ixtiyoriy)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Root loggerga yuqoriga tarqalishini oldini olish
    logger.propagate = False

    return logger
