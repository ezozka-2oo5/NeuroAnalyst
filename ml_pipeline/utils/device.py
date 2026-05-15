"""
ml_pipeline/utils/device.py
============================
Mavjud hisoblash qurilmasini (CPU/GPU/MPS) avtomatik aniqlash.
"""

from __future__ import annotations

import platform


def get_compute_device() -> "torch.device":
    """
    Eng yaxshi mavjud qurilmani qaytaradi:
      - NVIDIA CUDA GPU
      - Apple Silicon MPS (M1/M2/M3)
      - CPU (fallback)

    Returns:
        torch.device object.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
            props = torch.cuda.get_device_properties(device)
            print(
                f"[Device] NVIDIA CUDA GPU topildi: {props.name} "
                f"| {props.total_memory // 1024 ** 3} GB VRAM"
            )
            return device

        # Apple Silicon (M1/M2/M3) uchun Metal Performance Shaders
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[Device] Apple MPS topildi (Metal Performance Shaders)")
            return device

        print("[Device] GPU topilmadi — CPU ishlatiladi")
        return torch.device("cpu")

    except ImportError:
        raise RuntimeError(
            "PyTorch o'rnatilmagan. 'pip install torch' bilan o'rnating."
        )
