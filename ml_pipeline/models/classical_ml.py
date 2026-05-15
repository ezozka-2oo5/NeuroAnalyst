"""
ml_pipeline/models/classical_ml.py
=====================================
Scikit-learn asosidagi klassik ML modellari:
XGBoost, Random Forest, SVM.

Deep Learning uchun GPU shart, ammo bu modellar
CPU da ham tez ishlaydi va kichikroq datasetlarda
ko'pincha DL dan ustun keladi.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from ..utils.config import ModelConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ClassicalEEGClassifier:
    """
    XGBoost, Random Forest yoki SVM bilan EEG tasniflash.

    Bu klass scikit-learn interfeysi bilan mos:
    fit() → predict() → predict_proba()

    Qo'llanilishi::

        clf = ClassicalEEGClassifier(cfg, estimator="xgboost")
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)    # (N, n_classes)
        preds = clf.predict(X_test)          # (N,)
        clf.save("model.pkl")
    """

    SUPPORTED = ("xgboost", "random_forest", "gradient_boosting", "svm")

    def __init__(
        self,
        config: ModelConfig,
        estimator: Optional[str] = None,
    ) -> None:
        self.config = config
        self.estimator_name = estimator or config.classical_estimator
        self.label_encoder = LabelEncoder()
        self.model: Any = None
        self.class_weights_: Optional[np.ndarray] = None
        self._is_fitted = False

        if self.estimator_name not in self.SUPPORTED:
            raise ValueError(
                f"Noto'g'ri estimator: '{self.estimator_name}'. "
                f"Tanlang: {self.SUPPORTED}"
            )

    def _build_model(self, class_weight_dict: Optional[dict] = None) -> Any:
        """Tanlangan estimator obyektini yaratadi."""

        if self.estimator_name == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError:
                raise ImportError(
                    "XGBoost o'rnatilmagan. pip install xgboost"
                )
            scale_pos_weight = None
            if class_weight_dict and len(class_weight_dict) == 2:
                # Binary tasniflash uchun pos/neg nisbati
                scale_pos_weight = class_weight_dict.get(0, 1) / max(
                    class_weight_dict.get(1, 1), 1e-9
                )
            return XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                use_label_encoder=False,
                tree_method="hist",      # Tez hisoblash
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            )

        elif self.estimator_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=None,        # To'liq o'sgan daraxtlar
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight=class_weight_dict or "balanced",
                oob_score=True,        # Out-of-bag validation
                n_jobs=-1,
                random_state=42,
                verbose=0,
            )

        elif self.estimator_name == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                verbose=0,
            )

        elif self.estimator_name == "svm":
            return SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight=class_weight_dict or "balanced",
                probability=True,    # predict_proba uchun kerak
                random_state=42,
                cache_size=2000,     # MB, katta dataset uchun kerak
                max_iter=5000,
            )

        raise ValueError(f"Noma'lum estimator: {self.estimator_name}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ClassicalEEGClassifier":
        """
        Modelni o'qitadi.

        Args:
            X: (N, n_features) feature matrix.
            y: (N,) label array.
            X_val: Validation features (XGBoost early stopping uchun).
            y_val: Validation labels.

        Returns:
            self.
        """
        y_enc = self.label_encoder.fit_transform(y)
        classes = np.unique(y_enc)

        # Class weights
        weights = compute_class_weight("balanced", classes=classes, y=y_enc)
        self.class_weights_ = weights
        class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

        logger.info(
            f"{self.estimator_name} o'qitish boshlandi. "
            f"Train size: {len(X)}, Features: {X.shape[1]}, "
            f"Class weights: {class_weight_dict}"
        )

        self.model = self._build_model(class_weight_dict)

        # XGBoost uchun early stopping
        if self.estimator_name == "xgboost" and X_val is not None and y_val is not None:
            y_val_enc = self.label_encoder.transform(y_val)
            eval_set = [(X_val, y_val_enc)]
            fit_params = {
                "eval_set": eval_set,
                "verbose": False,
            }
            try:
                self.model.set_params(early_stopping_rounds=30)
            except Exception:
                pass
            self.model.fit(X, y_enc, **fit_params)
        else:
            self.model.fit(X, y_enc)

        self._is_fitted = True

        # Muhimlik (feature importance)
        if hasattr(self.model, "feature_importances_"):
            top_features = np.argsort(self.model.feature_importances_)[::-1][:10]
            logger.info(f"Top 10 muhim feature indekslari: {top_features.tolist()}")

        if self.estimator_name == "random_forest" and hasattr(self.model, "oob_score_"):
            logger.info(f"Random Forest OOB Score: {self.model.oob_score_:.4f}")

        logger.info(f"{self.estimator_name} o'qitish tugadi.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (N, n_features) float32.

        Returns:
            (N,) original label array.
        """
        self._check_fitted()
        y_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Har bir class uchun ehtimollik.

        Args:
            X: (N, n_features) float32.

        Returns:
            (N, n_classes) float32.
        """
        self._check_fitted()
        return self.model.predict_proba(X).astype(np.float32)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Feature importancelari (tree-based modellar uchun)."""
        self._check_fitted()
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None

    def save(self, path: str | Path) -> None:
        """Modelni diskka saqlaydi."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "model": self.model,
                    "label_encoder": self.label_encoder,
                    "estimator_name": self.estimator_name,
                    "class_weights": self.class_weights_,
                },
                fh,
            )
        logger.info(f"Model saqlandi: {path}")

    @classmethod
    def load(cls, path: str | Path, config: Optional[ModelConfig] = None) -> "ClassicalEEGClassifier":
        """Diskdan modelni yuklaydi."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)

        if config is None:
            config = ModelConfig()

        obj = cls(config, estimator=state["estimator_name"])
        obj.model = state["model"]
        obj.label_encoder = state["label_encoder"]
        obj.class_weights_ = state.get("class_weights")
        obj._is_fitted = True
        logger.info(f"Model yuklandi: {path}")
        return obj

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("fit() ni avval chaqiring.")
