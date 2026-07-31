"""
model.py -- the PREDICT layer.

A traditional fraud model does one thing: take a transaction, output a
fraud probability. That's the entire job of this module -- everything
about what to *do* with that probability lives in decision.py.

Primary backend: XGBoost (matches the reference architecture this project
is based on). Fallback backend: scikit-learn's GradientBoostingClassifier,
used automatically if xgboost isn't installed, so the project still runs
in environments where it can't be pip-installed. Both are gradient-boosted
tree ensembles, so downstream SHAP explanation code works against either.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import xgboost as xgb
    _BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _BACKEND = "sklearn_gbc"

FEATURE_COLUMNS = [f"V{i+1}" for i in range(28)] + ["Amount"]


@dataclass
class TrainResult:
    backend: str
    roc_auc: float
    pr_auc: float
    n_train: int
    n_test: int
    n_fraud_test: int


class FraudModel:
    """Thin, backend-agnostic wrapper around a gradient-boosted classifier."""

    def __init__(self, model=None, backend: Optional[str] = None):
        self.model = model
        self.backend = backend or _BACKEND

    # ---------------------------------------------------------------- train
    def fit(self, df: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> TrainResult:
        X = df[FEATURE_COLUMNS]
        y = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        if self.backend == "xgboost":
            pos = int(y_train.sum())
            neg = int(len(y_train) - pos)
            scale_pos_weight = neg / max(pos, 1)
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                random_state=seed,
                n_jobs=-1,
            )
            self.model.fit(X_train, y_train)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            # GradientBoostingClassifier has no class_weight param; upsample
            # the minority class instead to approximate scale_pos_weight.
            fraud_idx = y_train[y_train == 1].index
            repeat = max(int(len(y_train[y_train == 0]) / max(len(fraud_idx), 1) / 8), 1)
            X_train_bal = pd.concat([X_train, *([X_train.loc[fraud_idx]] * repeat)])
            y_train_bal = pd.concat([y_train, *([y_train.loc[fraud_idx]] * repeat)])
            self.model = GradientBoostingClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.9,
                random_state=seed,
            )
            self.model.fit(X_train_bal, y_train_bal)

        proba_test = self.predict_proba(X_test)
        result = TrainResult(
            backend=self.backend,
            roc_auc=float(roc_auc_score(y_test, proba_test)),
            pr_auc=float(average_precision_score(y_test, proba_test)),
            n_train=len(X_train),
            n_test=len(X_test),
            n_fraud_test=int(y_test.sum()),
        )
        # stash test split for downstream cost analysis / demo
        self._X_test, self._y_test, self._proba_test = X_test, y_test, proba_test
        return result

    # -------------------------------------------------------------- predict
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[FEATURE_COLUMNS] if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------ persist
    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "backend": self.backend}, f)

    @classmethod
    def load(cls, path: str) -> "FraudModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return cls(model=obj["model"], backend=obj["backend"])
