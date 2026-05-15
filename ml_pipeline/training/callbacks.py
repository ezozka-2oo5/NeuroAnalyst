"""
ml_pipeline/training/callbacks.py
=====================================
O'qitish jarayoni uchun callback moduli.

EarlyStopping, ModelCheckpoint va LearningRateScheduler klasslari
PyTorch trainer bilan integratsiya qilingan.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """
    Validation metrikasi yaxshilanmaganida o'qitishni to'xtatadi.

    Overfitting oldini olish uchun asosiy qurol.
    Best model holatini avtomatik saqlaydi.

    Qo'llanilishi::

        es = EarlyStopping(patience=15, metric="val_f1", mode="max")
        for epoch in range(max_epochs):
            ...
            if es.step(val_f1_score, model):
                break  # O'qitish to'xtatildi
    """

    def __init__(
        self,
        patience: int = 15,
        metric: str = "val_f1",
        mode: str = "max",
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
    ) -> None:
        """
        Args:
            patience: Necha epoch yaxshilanmasadan keyin to'xtatish.
            metric: Kuzatiladigan metrika nomi (faqat log uchun).
            mode: "max" = katta yaxshiroq, "min" = kichik yaxshiroq.
            min_delta: Yaxshilanish deb hisoblanishi uchun minimal o'zgarish.
            restore_best_weights: To'xtatilganda eng yaxshi model holati tiklansin.
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_value: Optional[float] = None
        self.best_weights: Optional[Dict[str, Any]] = None
        self.counter = 0
        self.stopped_epoch = 0

    def step(self, current_value: float, model: nn.Module) -> bool:
        """
        Yangi metrika qiymatini tekshiradi.

        Args:
            current_value: Hozirgi epoch ning metrika qiymati.
            model: Hozirgi model (best weights uchun).

        Returns:
            True = o'qitishni to'xtatish vaqti. False = davom etish.
        """
        improved = self._is_improved(current_value)

        if improved:
            self.best_value = current_value
            self.counter = 0
            # Eng yaxshi holatni xotirada saqlash
            import copy
            self.best_weights = copy.deepcopy(model.state_dict())
            logger.debug(f"EarlyStopping: yangi best {self.metric}={current_value:.4f}")
        else:
            self.counter += 1
            logger.debug(
                f"EarlyStopping: yaxshilanmadi. "
                f"Counter: {self.counter}/{self.patience} "
                f"(best={self.best_value:.4f})"
            )

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info(
                    f"EarlyStopping: {self.patience} epoch yaxshilanmadi. "
                    f"Best {self.metric}={self.best_value:.4f}. Best weights tiklandi."
                )
            else:
                logger.info(
                    f"EarlyStopping: {self.patience} epoch yaxshilanmadi. O'qitish to'xtatildi."
                )
            return True  # To'xtatish

        return False

    def _is_improved(self, value: float) -> bool:
        """Yangi qiymat yaxshilanganmi."""
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        else:  # mode == "min"
            return value < self.best_value - self.min_delta

    @property
    def best(self) -> Optional[float]:
        return self.best_value


class ModelCheckpoint:
    """
    Har bir epoch da model holatini diskka saqlaydi.

    save_best_only=True bo'lsa faqat yaxshilangan modellar saqlanadi,
    disk makonini tejaydi.

    Qo'llanilishi::

        ckpt = ModelCheckpoint(dirpath="checkpoints/", metric="val_f1", mode="max")
        for epoch in range(max_epochs):
            ckpt.save(model, optimizer, epoch, current_val_f1)
    """

    def __init__(
        self,
        dirpath: str | Path,
        metric: str = "val_f1",
        mode: str = "max",
        save_best_only: bool = True,
        save_last: bool = True,
        prefix: str = "eeg_model",
    ) -> None:
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.prefix = prefix
        self.best_value: Optional[float] = None

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric_value: float,
        extra_info: Optional[dict] = None,
    ) -> Optional[Path]:
        """
        Model holatini saqlaydi.

        Args:
            model: PyTorch modeli.
            optimizer: Optimizer (davom ettirish uchun).
            epoch: Hozirgi epoch raqami.
            metric_value: Hozirgi metrika qiymati.
            extra_info: Qo'shimcha ma'lumotlar (train loss va h.k.).

        Returns:
            Saqlangan fayl yo'li yoki None (saqlanmasa).
        """
        should_save = not self.save_best_only or self._is_best(metric_value)

        if should_save:
            self.best_value = metric_value
            path = self.dirpath / f"{self.prefix}_best.pt"
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metric": metric_value,
                "metric_name": self.metric,
            }
            if extra_info:
                state.update(extra_info)
            torch.save(state, path)
            logger.info(f"Checkpoint saqlandi: {path} ({self.metric}={metric_value:.4f})")

            # Har bir epoch numbered checkpoint
            if not self.save_best_only:
                numbered_path = self.dirpath / f"{self.prefix}_epoch_{epoch:03d}.pt"
                torch.save(state, numbered_path)

            return path

        # Oxirgi model (davom ettirish uchun)
        if self.save_last:
            last_path = self.dirpath / f"{self.prefix}_last.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metric": metric_value,
                },
                last_path,
            )

        return None

    @staticmethod
    def load(path: str | Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
        """
        Checkpoint dan model va optimizer holatini yuklaydi.

        Args:
            path: Checkpoint fayl yo'li.
            model: Holat yuklanadigan model.
            optimizer: Holat yuklanadigan optimizer (ixtiyoriy).

        Returns:
            Checkpoint state dict (epoch, metric va h.k.).
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        if optimizer and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        epoch = state.get("epoch", 0)
        metric = state.get("metric", 0.0)
        logger.info(f"Checkpoint yuklandi: {path} (epoch={epoch}, metric={metric:.4f})")
        return state

    def _is_best(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value


class LearningRateScheduler:
    """
    O'qitish jarayonida learning rate ni boshqaradi.

    Strategiyalar:
    - cosine: CosineAnnealingLR + warmup
    - step: StepLR
    - plateau: ReduceLROnPlateau (val loss asosida)
    - warmup_cosine: Linear warmup + cosine decay (eng popular)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        strategy: str = "cosine",
        max_epochs: int = 100,
        warmup_epochs: int = 5,
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        step_size: int = 30,
        step_gamma: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.strategy = strategy
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self._internal_scheduler = self._build(step_size, step_gamma)

    def _build(self, step_size: int, step_gamma: float):
        """Ichki PyTorch scheduler ni yaratadi."""
        if self.strategy == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epochs - self.warmup_epochs,
                eta_min=self.min_lr,
            )
        elif self.strategy == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=step_gamma
            )
        elif self.strategy == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=self.min_lr,
            )
        return None

    def step(self, epoch: int, val_metric: Optional[float] = None) -> float:
        """
        Epoch oxirida LR ni yangilaydi.

        Args:
            epoch: Hozirgi epoch (0-indexed).
            val_metric: Plateau scheduler uchun kerak.

        Returns:
            Yangilangan learning rate.
        """
        # Warmup bosqichi
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / max(self.warmup_epochs, 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.base_lr * warmup_factor
            return self.base_lr * warmup_factor

        # Asosiy scheduler
        if self._internal_scheduler is not None:
            if self.strategy == "plateau" and val_metric is not None:
                self._internal_scheduler.step(val_metric)
            else:
                self._internal_scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]
        return current_lr
