"""
backend/utils/logger.py
========================
Logging konfiguratsiyasi va yordamchi funksiyalar.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Logger yaratish yoki mavjud loggerni qaytarish.
    
    Args:
        name: Logger nomi (odatda __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Agar allaqachon handler qo'shilgan bo'lsa, qaytaramiz
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger

# Made with Bob
