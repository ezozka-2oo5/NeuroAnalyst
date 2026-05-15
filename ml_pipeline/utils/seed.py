"""
ml_pipeline/utils/seed.py
==========================
Tajribaning takrorlanishini ta'minlovchi seed o'rnatish.
Bir xil seed → bir xil natija.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """
    Python, NumPy va PyTorch uchun random seedni o'rnatadi.

    Args:
        seed: Takrorlanishni ta'minlovchi son.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch o'rnatilgan bo'lsa
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministik operatsiyalar uchun (bir oz sekinroq ishlaydi)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
