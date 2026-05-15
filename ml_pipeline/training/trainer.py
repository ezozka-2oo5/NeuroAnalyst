"""
ml_pipeline/training/trainer.py
==================================
PyTorch EEG model o'qitishni boshqaruvchi asosiy Trainer klassi.

Funksiyalar:
- Train/Val loop boshqaruvi
- Loss hisoblash (class-weighted cross-entropy)
- Early stopping va checkpoint bilan integratsiya
- Metrics logging (epoch bo'yicha)
- Mixed Precision (AMP) qo'llab-quvvatlash — GPU da 2x tezroq
- Gradient clipping — portlovchi gradient muammosini oldini olish
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from ..utils.config import TrainingConfig
from ..utils.device import get_compute_device
from ..utils.logger import get_logger
from ..utils.seed import set_global_seed

logger = get_logger(__name__)


class EEGModelTrainer:
    """
    EEG Deep Learning modellarini o'qituvchi klass.

    Qo'llanilishi::

        trainer = EEGModelTrainer(cfg, model)
        history = trainer.fit(X_train, y_train, X_val, y_val)
        metrics = trainer.evaluate(X_test, y_test)
        trainer.save_model("model_final.pt")
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        checkpoint_dir: Optional[str | Path] = None,
        log_dir: Optional[str | Path] = None,
    ) -> None:
        self.config = config
        self.device = get_compute_device()
        self.model = model.to(self.device)

        set_global_seed(config.seed)

        # Checkpoint va log joylar
        self.checkpoint_dir = Path(checkpoint_dir or config.checkpoint_dir)
        self.log_dir = Path(log_dir or config.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            metric=config.early_stopping_metric,
            mode=config.early_stopping_mode,
        )
        self.checkpointer = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            metric=config.checkpoint_metric,
            mode="max",
            save_best_only=config.save_best_only,
        )

        # Mixed Precision Scaler (GPU da ishlasa)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # O'qitish tarixi
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
            "lr": [],
        }

        logger.info(
            f"EEGModelTrainer tayyor: device={self.device}, "
            f"AMP={'yoqilgan' if self.use_amp else 'o\'chirilgan'}, "
            f"model={type(model).__name__}"
        )

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Modelni o'qitadi.

        Args:
            X_train: (N_train, n_channels, n_timepoints) float32.
            y_train: (N_train,) int64.
            X_val: (N_val, n_channels, n_timepoints) float32.
            y_val: (N_val,) int64.
            class_weights: Har bir class uchun og'irlik (imbalance uchun).

        Returns:
            history: Train/val loss, accuracy, F1 tarixi.
        """
        # Optimizer
        optimizer = self._build_optimizer()

        # Loss funksiyasi
        loss_fn = self._build_loss(y_train, class_weights)

        # LR Scheduler
        lr_scheduler = LearningRateScheduler(
            optimizer,
            strategy=self.config.scheduler or "cosine",
            max_epochs=self.config.epochs,
            warmup_epochs=self.config.scheduler_warmup_epochs,
            base_lr=self.config.learning_rate,
            min_lr=self.config.scheduler_min_lr,
        )

        # DataLoader
        train_loader = self._make_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._make_dataloader(X_val, y_val, shuffle=False)

        logger.info(
            f"O'qitish boshlandi: {self.config.epochs} epoch, "
            f"batch_size={self.config.batch_size}, "
            f"train={len(X_train)}, val={len(X_val)}"
        )

        best_val_f1 = 0.0
        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train bir epoch
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, loss_fn)

            # Validation
            val_loss, val_acc, val_f1 = self._val_epoch(val_loader, loss_fn)

            # LR yangilash
            current_lr = lr_scheduler.step(epoch, val_f1)

            # Checkpoint saqlash
            self.checkpointer.save(self.model, optimizer, epoch, val_f1)

            # Tarix yozish
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch [{epoch + 1:3d}/{self.config.epochs}] "
                f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f} | "
                f"lr={current_lr:.2e} | {epoch_time:.1f}s"
            )

            # Early stopping
            if self.early_stopping.step(val_f1, self.model):
                logger.info(f"Early stopping: epoch {epoch + 1}")
                break

        total_time = time.time() - start_time
        logger.info(
            f"O'qitish tugadi. Jami vaqt: {total_time / 60:.1f} daqiqa. "
            f"Best val_f1={self.early_stopping.best:.4f}"
        )
        return self.history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Test to'plamida model sifatini baholaydi.

        Args:
            X_test: (N_test, n_channels, n_timepoints) float32.
            y_test: (N_test,) int64.

        Returns:
            Metrics dict: accuracy, f1, precision, recall, auroc.
        """
        from ..evaluation.metrics import EEGClassificationMetrics

        self.model.eval()
        all_probs: List[np.ndarray] = []
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        test_loader = self._make_dataloader(X_test, y_test, shuffle=False)

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    logits = self.model(X_batch)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(y_batch.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        y_prob = np.concatenate(all_probs)

        metrics_calc = EEGClassificationMetrics()
        metrics = metrics_calc.compute(y_true, y_pred, y_prob)
        metrics_calc.print_report(metrics)
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Yangi ma'lumotlar uchun class ehtimolliklarini qaytaradi.

        Args:
            X: (N, n_channels, n_timepoints) float32.

        Returns:
            (N, n_classes) float32.
        """
        self.model.eval()
        loader = self._make_dataloader(X, np.zeros(len(X), dtype=np.int64), shuffle=False)
        all_probs: List[np.ndarray] = []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    logits = self.model(X_batch)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs).astype(np.float32)

    def save_model(self, path: str | Path) -> None:
        """Modelni diskka saqlaydi."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_class": type(self.model).__name__,
                "history": self.history,
            },
            path,
        )
        logger.info(f"Model saqlandi: {path}")

    def load_best_model(self) -> None:
        """Eng yaxshi checkpoint ni yuklab oladi."""
        best_path = self.checkpoint_dir / "eeg_model_best.pt"
        if best_path.exists():
            ModelCheckpoint.load(best_path, self.model)
        else:
            logger.warning(f"Best checkpoint topilmadi: {best_path}")

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> Tuple[float, float]:
        """Bitta train epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Data augmentation
            if self.config.apply_augmentation:
                X_batch = self._augment(X_batch)

            # Forward pass (Mixed Precision)
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(X_batch)
                loss = loss_fn(logits, y_batch)

            # Backward pass
            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        return total_loss / max(total, 1), correct / max(total, 1)

    def _val_epoch(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Tuple[float, float, float]:
        """Bitta validation epoch."""
        from sklearn.metrics import f1_score

        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    logits = self.model(X_batch)
                    loss = loss_fn(logits, y_batch)

                total_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(y_batch.cpu().tolist())
                total += len(y_batch)

        acc = sum(p == t for p, t in zip(all_preds, all_targets)) / max(total, 1)
        f1 = float(
            f1_score(all_targets, all_preds, average="macro", zero_division=0)
        )
        avg_loss = total_loss / max(total, 1)
        return avg_loss, acc, f1

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Konfiguratsiyaga qarab optimizer yaratadi."""
        params = self.model.parameters()
        lr = self.config.learning_rate
        wd = self.config.weight_decay

        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        raise ValueError(f"Noma'lum optimizer: {self.config.optimizer}")

    def _build_loss(
        self,
        y_train: np.ndarray,
        class_weights: Optional[np.ndarray],
    ) -> nn.Module:
        """Class-weighted CrossEntropyLoss yaratadi."""
        if self.config.use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            if class_weights is None:
                class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            logger.info(f"Class weights: {class_weights.tolist()}")
            return nn.CrossEntropyLoss(weight=weights_tensor)
        return nn.CrossEntropyLoss()

    def _make_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = False,
    ) -> DataLoader:
        """PyTorch DataLoader yaratadi."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)

        sampler = None
        if shuffle and self.config.oversample_minority:
            # WeightedRandomSampler bilan minority class ni ko'paytirish
            class_counts = np.bincount(y)
            class_weights = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights[y]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.float32),
                num_samples=len(y),
                replacement=True,
            )
            shuffle = False  # Sampler bilan shuffle bir vaqtda bo'lmaydi

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=min(4, torch.get_num_threads()),
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def _augment(self, X: torch.Tensor) -> torch.Tensor:
        """On-the-fly data augmentation."""
        cfg = self.config

        # Gaussian noise
        if cfg.aug_noise_std > 0:
            noise = torch.randn_like(X) * cfg.aug_noise_std
            X = X + noise

        # Amplitude scaling
        low, high = cfg.aug_amplitude_scale
        scale = torch.empty(X.size(0), 1, 1).uniform_(low, high).to(X.device)
        X = X * scale

        # Time shift (padding bilan)
        max_shift = cfg.aug_time_shift_max
        if max_shift > 0:
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            if shift > 0:
                X = torch.cat([X[:, :, shift:], torch.zeros_like(X[:, :, :shift])], dim=2)
            elif shift < 0:
                s = -shift
                X = torch.cat([torch.zeros_like(X[:, :, :s]), X[:, :, :-s]], dim=2)

        return X
