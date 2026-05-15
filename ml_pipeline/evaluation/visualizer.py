"""
ml_pipeline/evaluation/visualizer.py
========================================
O'qitish natijalari va model sifatini vizualizatsiya qiluvchi klass.

Grafiklar:
- Training curve (loss, accuracy, F1)
- Confusion matrix heatmap
- ROC curve (AUROC bilan)
- Precision-Recall curve
- Feature importance bar chart
- EEG epoch + prediction vizualizatsiyasi
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Headless server uchun (GUI talab qilmaydi)
import matplotlib.pyplot as plt
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationVisualizer:
    """
    Model baholash natijalarini vizualizatsiya qiluvchi klass.

    Qo'llanilishi::

        viz = EvaluationVisualizer(output_dir="ml_pipeline/plots")
        viz.plot_training_curves(history)
        viz.plot_confusion_matrix(y_true, y_pred)
        viz.plot_roc_curve(y_true, y_prob)
    """

    def __init__(
        self,
        output_dir: str | Path = "ml_pipeline/plots",
        dpi: int = 150,
        style: str = "dark_background",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("default")

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_name: str = "training_curves.png",
    ) -> Path:
        """
        O'qitish tarixi (loss, accuracy, F1, LR) grafiklari.

        Args:
            history: trainer.fit() natijasi dict.
            save_name: Saqlash fayl nomi.

        Returns:
            Saqlangan fayl yo'li.
        """
        epochs_range = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("O'qitish tarixi", fontsize=16, fontweight="bold", color="white")

        # Loss
        ax = axes[0, 0]
        ax.plot(epochs_range, history["train_loss"], label="Train Loss", color="#22d3ee", linewidth=2)
        ax.plot(epochs_range, history["val_loss"], label="Val Loss", color="#f97316", linewidth=2)
        ax.set_title("Loss", color="white")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

        # Accuracy
        ax = axes[0, 1]
        ax.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#22d3ee", linewidth=2)
        ax.plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Val Acc", color="#f97316", linewidth=2)
        ax.set_title("Accuracy (%)", color="white")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        ax.grid(alpha=0.3)

        # F1
        ax = axes[1, 0]
        ax.plot(epochs_range, history["val_f1"], label="Val Macro F1", color="#64ffda", linewidth=2)
        best_f1_epoch = int(np.argmax(history["val_f1"])) + 1
        best_f1 = max(history["val_f1"])
        ax.axvline(best_f1_epoch, color="#f97316", linestyle="--", alpha=0.7, label=f"Best: {best_f1:.4f}")
        ax.set_title("Val Macro F1", color="white")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.legend()
        ax.grid(alpha=0.3)

        # LR
        ax = axes[1, 1]
        ax.semilogy(epochs_range, history["lr"], color="#a78bfa", linewidth=2)
        ax.set_title("Learning Rate", color="white")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR (log scale)")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="black")
        plt.close()
        logger.info(f"Training curves saqlandi: {save_path}")
        return save_path

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: str = "confusion_matrix.png",
    ) -> Path:
        """
        Confusion matrix heatmap.

        Args:
            y_true: Haqiqiy labellar.
            y_pred: Model bashoratlari.
            class_names: Class nomlari.
            save_name: Fayl nomi.
        """
        from sklearn.metrics import confusion_matrix
        import matplotlib.colors as mcolors

        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        # Normalizatsiya qilingan va xom CM yon-yon
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Confusion Matrix", fontsize=16, fontweight="bold", color="white")

        for ax, (data, fmt, title) in zip(
            axes,
            [
                (cm, "d", "Xom son"),
                (cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1), ".2f", "Normallashtirilgan"),
            ],
        ):
            im = ax.imshow(data, interpolation="nearest", cmap="Blues")
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
            ax.set_xlabel("Bashorat", color="white")
            ax.set_ylabel("Haqiqiy", color="white")
            ax.set_title(title, color="white")

            thresh = data.max() / 2.0
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(
                        j, i, format(data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if data[i, j] < thresh else "black",
                        fontsize=11,
                    )

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="black")
        plt.close()
        logger.info(f"Confusion matrix saqlandi: {save_path}")
        return save_path

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: str = "roc_curve.png",
    ) -> Path:
        """
        ROC egri chizig'i (AUROC bilan).

        Args:
            y_true: Haqiqiy labellar.
            y_prob: (N, n_classes) ehtimolliklar.
            class_names: Class nomlari.
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        n_classes = y_prob.shape[1]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_facecolor("#0a192f")
        colors = ["#22d3ee", "#64ffda", "#f97316", "#a78bfa", "#fbbf24"]

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[0], lw=2, label=f"AUC = {roc_auc:.4f}")
        else:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            for i, (cls, color) in enumerate(zip(class_names, colors)):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} AUC={roc_auc:.4f}")

        ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5, label="Tasodifiy model")
        ax.set_xlabel("False Positive Rate", color="white")
        ax.set_ylabel("True Positive Rate", color="white")
        ax.set_title("ROC Egri Chizig'i", color="white", fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
        ax.tick_params(colors="white")

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="#061a2f")
        plt.close()
        logger.info(f"ROC curve saqlandi: {save_path}")
        return save_path

    def plot_feature_importances(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        top_n: int = 20,
        save_name: str = "feature_importance.png",
    ) -> Path:
        """
        Feature importancelari bar chart.

        Args:
            importances: Feature importance array.
            feature_names: Feature nomlari.
            top_n: Faqat eng muhim N ta ko'rsatilsin.
        """
        idx = np.argsort(importances)[::-1][:top_n]
        top_imps = importances[idx]
        top_names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.set_facecolor("#0a192f")
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
        bars = ax.barh(range(len(top_imps))[::-1], top_imps, color=colors)
        ax.set_yticks(range(len(top_imps))[::-1])
        ax.set_yticklabels(top_names, fontsize=8, color="white")
        ax.set_xlabel("Muhimlik darajasi", color="white")
        ax.set_title(f"Top {top_n} Feature Importances", color="white", fontsize=14)
        ax.tick_params(colors="white")
        ax.grid(axis="x", alpha=0.2)

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="#061a2f")
        plt.close()
        logger.info(f"Feature importance saqlandi: {save_path}")
        return save_path
