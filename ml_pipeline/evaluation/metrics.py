"""
ml_pipeline/evaluation/metrics.py
=====================================
EEG tasniflash modellarini baholash uchun to'liq metrikalar.

Tibbiy tashxisda faqat Accuracy yetarli emas.
Seizure kabi noyob hodisalar uchun:
- Recall (Sensitivity) — kasalni o'tkazib yubormaslik
- Specificity — sog'lom odamni noto'g'ri kasallamaylik
- AUROC — threshold ga bog'liq bo'lmagan umumiy sifat
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EEGClassificationMetrics:
    """
    EEG tasniflash natijalari uchun to'liq metrika hisoblash.

    Qo'llanilishi::

        metrics = EEGClassificationMetrics()
        result = metrics.compute(y_true, y_pred, y_prob)
        metrics.print_report(result)
    """

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Barcha metrikalarni hisoblaydi.

        Args:
            y_true: Haqiqiy labellar (N,).
            y_pred: Model bashoratlari (N,).
            y_prob: Class ehtimolliklari (N, n_classes). AUROC uchun kerak.
            class_names: Har bir class uchun nom (ixtiyoriy).

        Returns:
            Dict: accuracy, macro_f1, weighted_f1, precision, recall,
                  specificity, auroc, cohen_kappa, mcc va h.k.
        """
        from sklearn.metrics import (
            accuracy_score,
            cohen_kappa_score,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics: Dict[str, float] = {}

        # ── Asosiy metrikalar ─────────────────────────────────────
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))  # Matthews Correlation Coefficient

        # ── Confusion matrix ──────────────────────────────────────
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm  # type: ignore

        n_classes = len(np.unique(y_true))

        # ── Binary klaslash uchun qo'shimcha metrikalar ───────────
        if n_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / max(tp + fn, 1)   # Recall
            specificity = tn / max(tn + fp, 1)
            ppv = tp / max(tp + fp, 1)            # Positive Predictive Value
            npv = tn / max(tn + fn, 1)            # Negative Predictive Value
            balanced_acc = (sensitivity + specificity) / 2.0

            metrics["sensitivity"] = float(sensitivity)
            metrics["specificity"] = float(specificity)
            metrics["ppv"] = float(ppv)
            metrics["npv"] = float(npv)
            metrics["balanced_accuracy"] = float(balanced_acc)

        # ── AUROC ────────────────────────────────────────────────
        if y_prob is not None:
            try:
                if n_classes == 2:
                    auroc = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    auroc = float(
                        roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                    )
                metrics["auroc"] = auroc
            except ValueError as exc:
                logger.warning(f"AUROC hisoblanmadi: {exc}")
                metrics["auroc"] = 0.0

        # ── Har bir class uchun alohida metrikalar ───────────────
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)

        classes = np.unique(y_true)
        for i, cls in enumerate(classes):
            cls_name = class_names[i] if class_names and i < len(class_names) else f"class_{cls}"
            metrics[f"{cls_name}_f1"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0
            metrics[f"{cls_name}_precision"] = float(per_class_prec[i]) if i < len(per_class_prec) else 0.0
            metrics[f"{cls_name}_recall"] = float(per_class_rec[i]) if i < len(per_class_rec) else 0.0

        return metrics

    def print_report(self, metrics: Dict, title: str = "MODEL BAHOLASH NATIJASI") -> None:
        """Metrikalarni chiroyli formatda chiqaradi."""
        from sklearn.metrics import classification_report

        separator = "─" * 60
        print(f"\n{separator}")
        print(f"  {title}")
        print(separator)

        key_metrics = [
            ("Accuracy", "accuracy"),
            ("Macro F1", "macro_f1"),
            ("Weighted F1", "weighted_f1"),
            ("Macro Precision", "macro_precision"),
            ("Macro Recall", "macro_recall"),
            ("Cohen Kappa", "cohen_kappa"),
            ("MCC", "mcc"),
            ("AUROC", "auroc"),
            ("Sensitivity", "sensitivity"),
            ("Specificity", "specificity"),
            ("Balanced Accuracy", "balanced_accuracy"),
        ]

        for name, key in key_metrics:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, float):
                    print(f"  {name:<25}: {val:.4f}")

        print(separator)

        cm = metrics.get("confusion_matrix")
        if cm is not None and hasattr(cm, "tolist"):
            print(f"\n  Confusion Matrix:\n{np.array2string(cm, prefix='  ')}")
        print()
