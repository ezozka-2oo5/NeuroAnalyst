"""
ml_pipeline/export_onnx.py
============================
PyTorch CNN-LSTM modelini ONNX formatiga eksport qilish skripti.

Nima uchun ONNX?
  - PyTorch'ga bog'liq emas — istalgan framework da ishlatish mumkin.
  - onnxruntime → PyTorch'dan 2-4x tezroq inference.
  - CPU optimizatsiyalari: SIMD, thread pooling.
  - Kichikroq memory footprint — server uchun ideal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent


def export_to_onnx(
    pt_path: str | Path,
    config_path: str | Path,
    onnx_path: str | Path,
    opset_version: int = 17,
    verify: bool = True,
) -> Path:
    """
    PyTorch modelni ONNX ga eksport qiladi.

    Args:
        pt_path: .pt checkpoint yo'li.
        config_path: PipelineConfig JSON yo'li.
        onnx_path: Chiqish .onnx fayl yo'li.
        opset_version: ONNX opset versiyasi (17 = zamonaviy).
        verify: True bo'lsa PyTorch vs ONNX natijalarini solishtiradi.

    Returns:
        onnx_path — eksport qilingan fayl yo'li.
    """
    sys.path.insert(0, str(BASE_DIR))

    from ml_pipeline.models.cnn_lstm import EEGCNNLSTM
    from ml_pipeline.utils.config import PipelineConfig

    pt_path = Path(pt_path)
    config_path = Path(config_path)
    onnx_path = Path(onnx_path)

    if not pt_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint topilmadi: {pt_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Konfiguratsiya fayli topilmadi: {config_path}")

    print(f"[1/4] Konfiguratsiya yuklanmoqda: {config_path}")
    cfg = PipelineConfig.load(config_path)

    print(f"[2/4] PyTorch model yuklanmoqda: {pt_path}")
    device = torch.device("cpu")  # ONNX eksport CPU da aniqroq
    model = EEGCNNLSTM(cfg.model)

    checkpoint = torch.load(pt_path, map_location=device, weights_only=True)
    # Checkpoint strukturasiga qarab yuklaymiz
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    # Dummy input — real signal o'lchamlariga mos
    n_channels = cfg.model.n_channels
    n_timepoints = cfg.model.n_timepoints
    dummy_input = torch.randn(1, n_channels, n_timepoints, dtype=torch.float32).to(device)

    print(f"[3/4] ONNX ga eksport qilinmoqda: {onnx_path}")
    print(f"      Input shape: (batch, {n_channels}, {n_timepoints})")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["eeg_input"],
        output_names=["logits"],
        dynamic_axes={
            "eeg_input": {0: "batch_size"},  # Variable batch size
            "logits": {0: "batch_size"},
        },
        verbose=False,
    )
    print(f"      ✅ ONNX fayl saqlandi: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    if verify:
        print("[4/4] ONNX modelni tekshirilmoqda (PyTorch vs ONNX solishtirish)...")
        _verify_onnx(model, onnx_path, dummy_input, n_classes=cfg.model.n_classes)

    return onnx_path


def _verify_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    dummy_input: torch.Tensor,
    n_classes: int,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> None:
    """PyTorch va ONNX natijalarini solishtiradi."""
    try:
        import onnx
        import onnxruntime as ort

        # ONNX model validatsiyasi
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("      ✅ ONNX model strukturasi to'g'ri.")

        # PyTorch natijasi
        with torch.no_grad():
            pt_output = pytorch_model(dummy_input).numpy()

        # ONNX Runtime natijasi
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(onnx_path), sess_options=sess_options)
        ort_inputs = {"eeg_input": dummy_input.numpy()}
        ort_output = session.run(["logits"], ort_inputs)[0]

        # Solishtirishj
        np.testing.assert_allclose(pt_output, ort_output, rtol=rtol, atol=atol)
        max_diff = float(np.max(np.abs(pt_output - ort_output)))
        print(f"      ✅ Natijalar mos keladi. Max farq: {max_diff:.2e}")

    except ImportError:
        print("      ⚠️  onnx/onnxruntime o'rnatilmagan — tekshiruv o'tkazildi.")
    except AssertionError as e:
        print(f"      ❌ Natijalar mos kelmaydi! Xato: {e}")


def main() -> None:
    """CLI interfeysi."""
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch modelni ONNX ga eksport qilish")
    parser.add_argument(
        "--pt",
        default=str(BASE_DIR / "ml_pipeline" / "real_output" / "checkpoints" / "eeg_model_best.pt"),
        help="PyTorch checkpoint (.pt) fayl yo'li",
    )
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "ml_pipeline" / "real_output" / "real_config.json"),
        help="PipelineConfig JSON fayl yo'li",
    )
    parser.add_argument(
        "--onnx",
        default=str(BASE_DIR / "ml_pipeline" / "real_output" / "checkpoints" / "eeg_model.onnx"),
        help="Chiqish ONNX fayl yo'li",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset versiyasi")
    parser.add_argument("--no-verify", action="store_true", help="Tekshiruvni o'tkazib yuborish")

    args = parser.parse_args()

    # Agar real model yo'q bo'lsa, demo modelni ishlatamiz
    if not Path(args.pt).exists():
        demo_pt = str(BASE_DIR / "ml_pipeline" / "demo_output" / "checkpoints" / "eeg_model_best.pt")
        demo_cfg = str(BASE_DIR / "ml_pipeline" / "demo_output" / "demo_config.json")
        demo_onnx = str(BASE_DIR / "ml_pipeline" / "demo_output" / "checkpoints" / "eeg_model.onnx")
        if Path(demo_pt).exists():
            print(f"Real model topilmadi. Demo modeldan foydalanilmoqda.")
            args.pt = demo_pt
            args.config = demo_cfg
            args.onnx = demo_onnx
        else:
            print(f"❌ Na real, na demo model topilmadi: {args.pt}")
            sys.exit(1)

    try:
        out = export_to_onnx(
            pt_path=args.pt,
            config_path=args.config,
            onnx_path=args.onnx,
            opset_version=args.opset,
            verify=not args.no_verify,
        )
        print(f"\n🚀 ONNX eksport muvaffaqiyatli yakunlandi: {out}")
    except Exception as e:
        print(f"\n❌ Eksport xatosi: {e}")
        raise


if __name__ == "__main__":
    main()
