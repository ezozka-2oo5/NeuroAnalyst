"""
ml_pipeline/evaluation/report_generator.py
============================================
O'qitish natijalari uchun PDF va text formatida
professional klinik hisobot generatsiya qiluvchi klass.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MLReportGenerator:
    """
    O'qitish va baholash natijalari asosida hisobot yaratadi.

    Qo'llanilishi::

        reporter = MLReportGenerator(output_dir="ml_pipeline/reports")
        reporter.generate(
            dataset_name="CHB-MIT Seizure",
            model_name="CNN-BiLSTM",
            metrics=metrics_dict,
            history=training_history,
        )
    """

    def __init__(self, output_dir: str | Path = "ml_pipeline/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        dataset_name: str,
        model_name: str,
        metrics: Dict,
        history: Optional[Dict[str, List[float]]] = None,
        config_dict: Optional[dict] = None,
        fold_results: Optional[List[Dict]] = None,
        save_json: bool = True,
        save_txt: bool = True,
    ) -> Dict[str, Path]:
        """
        Hisobot yaratadi.

        Args:
            dataset_name: Dataset nomi.
            model_name: Model arxitekturasi nomi.
            metrics: EEGClassificationMetrics.compute() natijasi.
            history: O'qitish tarixi.
            config_dict: Konfiguratsiya dict (JSON saqlanadi).
            fold_results: Cross-validation natijalar ro'yxati.
            save_json: JSON formatda saqlash.
            save_txt: Text formatda saqlash.

        Returns:
            Saqlangan fayl yo'llari dict.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_name.replace(' ', '_')}_{timestamp}"

        saved_files: Dict[str, Path] = {}

        # JSON hisobot
        if save_json:
            json_path = self._save_json(
                base_name, dataset_name, model_name, metrics, history, config_dict, fold_results
            )
            saved_files["json"] = json_path

        # Text hisobot
        if save_txt:
            txt_path = self._save_txt(
                base_name, dataset_name, model_name, metrics, history, fold_results
            )
            saved_files["txt"] = txt_path

        logger.info(f"Hisobotlar saqlandi: {[str(p) for p in saved_files.values()]}")
        return saved_files

    # ──────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────

    def _save_json(
        self,
        base_name: str,
        dataset_name: str,
        model_name: str,
        metrics: Dict,
        history: Optional[Dict],
        config_dict: Optional[dict],
        fold_results: Optional[List[Dict]],
    ) -> Path:
        """JSON formatda to'liq ma'lumot saqlash."""
        report: dict = {
            "generated_at": datetime.now().isoformat(),
            "dataset": dataset_name,
            "model": model_name,
            "metrics": self._serialize_metrics(metrics),
            "history_summary": self._summarize_history(history) if history else None,
            "cross_validation_folds": fold_results,
            "config": config_dict,
        }

        path = self.output_dir / f"{base_name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        return path

    def _save_txt(
        self,
        base_name: str,
        dataset_name: str,
        model_name: str,
        metrics: Dict,
        history: Optional[Dict],
        fold_results: Optional[List[Dict]],
    ) -> Path:
        """Inson o'qiy oladigan text hisobot."""
        lines: List[str] = []
        sep = "═" * 70
        thin = "─" * 70

        lines.extend([
            sep,
            "  NEUROANALYST ML PIPELINE — O'QITISH NATIJASI HISOBOTI".center(70),
            sep,
            f"  Sana:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Dataset:     {dataset_name}",
            f"  Model:       {model_name}",
            thin,
            "",
            "  ASOSIY METRIKALAR",
            thin,
        ])

        key_metrics_display = [
            ("Accuracy", "accuracy"),
            ("Macro F1 Score", "macro_f1"),
            ("Weighted F1 Score", "weighted_f1"),
            ("Macro Precision", "macro_precision"),
            ("Macro Recall", "macro_recall"),
            ("AUROC", "auroc"),
            ("Cohen's Kappa", "cohen_kappa"),
            ("Matthews CC (MCC)", "mcc"),
            ("Sensitivity (Recall)", "sensitivity"),
            ("Specificity", "specificity"),
            ("Balanced Accuracy", "balanced_accuracy"),
        ]

        for display_name, key in key_metrics_display:
            if key in metrics and isinstance(metrics[key], (int, float)):
                lines.append(f"  {display_name:<30}: {metrics[key]:.4f}  ({metrics[key]*100:.2f}%)")

        if history:
            lines.extend([
                "",
                thin,
                "  O'QITISH XULOSASI",
                thin,
            ])
            summary = self._summarize_history(history)
            for k, v in summary.items():
                lines.append(f"  {k:<30}: {v}")

        if fold_results:
            lines.extend([
                "",
                thin,
                "  CROSS-VALIDATION NATIJALARI",
                thin,
            ])
            for fold in fold_results:
                fold_num = fold.get("fold", "?")
                fold_f1 = fold.get("val_f1", 0.0)
                fold_acc = fold.get("val_accuracy", 0.0)
                lines.append(f"  Fold {fold_num}: Accuracy={fold_acc:.4f}, F1={fold_f1:.4f}")

            all_f1s = [f.get("val_f1", 0.0) for f in fold_results]
            if all_f1s:
                lines.extend([
                    thin,
                    f"  O'rtacha F1: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}",
                ])

        lines.extend(["", sep, ""])

        path = self.output_dir / f"{base_name}.txt"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return path

    @staticmethod
    def _serialize_metrics(metrics: Dict) -> Dict:
        """JSON serialize qilish uchun metriklarni tozalash."""
        result: Dict = {}
        for k, v in metrics.items():
            if k == "confusion_matrix" and hasattr(v, "tolist"):
                result[k] = v.tolist()
            elif isinstance(v, (int, float, str, bool)):
                result[k] = v
            elif isinstance(v, np.floating):
                result[k] = float(v)
            elif isinstance(v, np.integer):
                result[k] = int(v)
        return result

    @staticmethod
    def _summarize_history(history: Dict[str, List[float]]) -> Dict[str, str]:
        """O'qitish tarixini qisqacha xulosa qiladi."""
        summary: Dict[str, str] = {}
        n_epochs = len(history.get("train_loss", []))
        summary["Jami epoch"] = str(n_epochs)

        if "val_f1" in history and history["val_f1"]:
            best_f1 = max(history["val_f1"])
            best_epoch = int(np.argmax(history["val_f1"])) + 1
            summary["Eng yaxshi val_f1"] = f"{best_f1:.4f} (epoch {best_epoch})"

        if "val_acc" in history and history["val_acc"]:
            best_acc = max(history["val_acc"])
            summary["Eng yaxshi val_accuracy"] = f"{best_acc:.4f}"

        if "val_loss" in history and history["val_loss"]:
            final_loss = history["val_loss"][-1]
            summary["Oxirgi val_loss"] = f"{final_loss:.4f}"

        if "lr" in history and history["lr"]:
            final_lr = history["lr"][-1]
            summary["Oxirgi learning rate"] = f"{final_lr:.2e}"

        return summary
