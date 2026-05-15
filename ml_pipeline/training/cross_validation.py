"""
ml_pipeline/training/cross_validation.py
==========================================
Subject-wise Cross-Validation.

EEG o'qitishda eng muhim muammo — bitta subjectning signallari
ham train, ham test to'plamida bo'lib qolishi.

Bu holda model kasallikni emas, bemorning individual EEG imzosini
(shaxsiy to'lqin shaklini) yodlaydi va yolg'on yuqori accuracy beradi.

Subject-wise CV bu muammoni hal qiladi:
har bir fold da to'liq subjectlar ajratiladi.
"""

from __future__ import annotations

from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SubjectWiseCrossValidator:
    """
    Subject-wise K-Fold Cross-Validation generator.

    Har bir foldda:
    - Train: K-1 ta subjectlar
    - Val: 1 ta subject
    - Test: Alohida held-out subjectlar (opsional)

    Qo'llanilishi::

        cv = SubjectWiseCrossValidator(n_splits=5, test_ratio=0.2, seed=42)
        for fold, (train_idx, val_idx, test_idx) in enumerate(cv.split(X, y, subjects)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            ...
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_ratio: float = 0.2,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        """
        Args:
            n_splits: K (koldlar soni).
            test_ratio: Held-out test subjectlar ulushi.
            seed: Tasodifiylik uchun.
            shuffle: Subjectlarni aralashtirsin.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2 for cross-validation.")
        self.n_splits = n_splits
        self.test_ratio = test_ratio
        self.seed = seed
        self.shuffle = shuffle

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Train/Val/Test indekslarini generatsiya qiladi.

        Args:
            X: (N, ...) feature yoki epoch array.
            y: (N,) label array.
            subject_ids: (N,) har bir epoch qaysi subjectga tegishli ekanini bildiruvchi array.

        Yields:
            (train_idx, val_idx, test_idx) numpy array lari.
        """
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)

        if n_subjects < self.n_splits + 1:
            logger.warning(
                f"Subject soni ({n_subjects}) < n_splits+1 ({self.n_splits + 1}). "
                f"n_splits={n_subjects - 1} ga tushirildi."
            )
            self.n_splits = max(1, n_subjects - 1)

        rng = np.random.default_rng(self.seed)
        subjects = unique_subjects.copy()
        if self.shuffle:
            rng.shuffle(subjects)

        # Held-out test subjectlar (o'qitishda hech qachon ko'rilmaydi)
        n_test_subjects = max(1, int(n_subjects * self.test_ratio))
        test_subjects = subjects[:n_test_subjects]
        cv_subjects = subjects[n_test_subjects:]

        test_idx = np.where(np.isin(subject_ids, test_subjects))[0]
        logger.info(
            f"Test subjectlari: {test_subjects.tolist()} "
            f"({len(test_idx)} epoch)"
        )

        # K-Fold bo'lish
        fold_size = max(1, len(cv_subjects) // self.n_splits)
        for fold in range(self.n_splits):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_splits - 1 else len(cv_subjects)
            val_subjects = cv_subjects[val_start:val_end]
            train_subjects = np.concatenate([
                cv_subjects[:val_start],
                cv_subjects[val_end:],
            ])

            train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
            val_idx = np.where(np.isin(subject_ids, val_subjects))[0]

            logger.info(
                f"Fold {fold + 1}/{self.n_splits}: "
                f"Train={len(train_subjects)} subject ({len(train_idx)} epoch) | "
                f"Val={val_subjects.tolist()} ({len(val_idx)} epoch)"
            )

            yield train_idx, val_idx, test_idx

    def get_test_indices(
        self,
        subject_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Faqat test indekslarini qaytaradi (split ni chaqirmasdan).
        """
        unique_subjects = np.unique(subject_ids)
        rng = np.random.default_rng(self.seed)
        subjects = unique_subjects.copy()
        if self.shuffle:
            rng.shuffle(subjects)

        n_test = max(1, int(len(subjects) * self.test_ratio))
        test_subjects = subjects[:n_test]
        return np.where(np.isin(subject_ids, test_subjects))[0]

    def fold_summary(self, subject_ids: np.ndarray, y: np.ndarray) -> List[Dict]:
        """
        Har bir fold uchun statistika qaytaradi (vizualizatsiya uchun).
        """
        summaries: List[Dict] = []
        for fold, (train_idx, val_idx, test_idx) in enumerate(
            self.split(np.zeros((len(subject_ids), 1)), y, subject_ids)
        ):
            unique, counts = np.unique(y[train_idx], return_counts=True)
            summaries.append({
                "fold": fold + 1,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "train_label_dist": dict(zip(unique.tolist(), counts.tolist())),
            })
        return summaries
