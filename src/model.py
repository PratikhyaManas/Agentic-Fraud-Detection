"""
Predict layer: XGBoost fraud probability model + SHAP explainer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from .data_generator import FEATURE_NAMES


class FraudModel:
    """XGBoost binary classifier that outputs P(fraud) and SHAP values."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.08,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
    ):
        self.feature_names = FEATURE_NAMES
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.explainer: Optional[shap.TreeExplainer] = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FraudModel":
        X = X[self.feature_names]
        if self.model.scale_pos_weight is None:
            # Auto-set to inverse class frequency
            pos = y.sum()
            neg = len(y) - pos
            self.model.scale_pos_weight = neg / max(pos, 1)

        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X = X[self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP values for the positive (fraud) class."""
        self._check_fitted()
        X = X[self.feature_names]
        sv = self.explainer.shap_values(X)
        # TreeExplainer for binary returns array of shape (n, n_features)
        # or list depending on version; normalise to 2-d array
        if isinstance(sv, list):
            sv = sv[1]
        return np.asarray(sv)

    def explain_transaction(
        self, X_row: pd.DataFrame, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k features driving the fraud score for a single row,
        with direction and magnitude.  Ready to feed into the Act layer.
        """
        self._check_fitted()
        assert len(X_row) == 1
        sv = self.shap_values(X_row)[0]
        values = X_row[self.feature_names].iloc[0].to_dict()

        impacts = []
        for name, shap_val in zip(self.feature_names, sv):
            impacts.append(
                {
                    "feature": name,
                    "value": float(values[name]),
                    "shap": float(shap_val),
                    "direction": "increases_fraud_risk"
                    if shap_val > 0
                    else "decreases_fraud_risk",
                    "abs_impact": abs(float(shap_val)),
                }
            )
        impacts.sort(key=lambda d: d["abs_impact"], reverse=True)
        return impacts[:top_k]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        proba = self.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
            "report": classification_report(y, preds, output_dict=True),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "xgb_model.joblib")
        # Persist feature list for safety
        (path / "feature_names.json").write_text(json.dumps(self.feature_names))

    def load(self, path: str | Path) -> "FraudModel":
        path = Path(path)
        self.model = joblib.load(path / "xgb_model.joblib")
        self.feature_names = json.loads((path / "feature_names.json").read_text())
        self.explainer = shap.TreeExplainer(self.model)
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")


def train_and_persist(
    train_df: pd.DataFrame,
    model_dir: str | Path = "models",
) -> Tuple[FraudModel, Dict[str, float]]:
    """Convenience helper used by the training script."""
    X = train_df[FEATURE_NAMES]
    y = train_df["is_fraud"]
    model = FraudModel()
    model.fit(X, y)
    metrics = model.evaluate(X, y)  # train metrics for sanity
    model.save(model_dir)
    return model, metrics