"""
explain.py -- feature attribution that feeds the ACT layer.

This module intentionally does NOT hand raw feature names (V1, V14, "Amount")
or SHAP magnitudes straight to the LLM. Two reasons, both learned the hard
way when building the system this is modeled on:

1. Raw PCA component names (V1..V28) are meaningless to a human reviewer --
   surfacing "V14 = -6.2" in a summary is technically transparent but
   practically useless jargon.
2. Handing the LLM raw numbers invites it to either parrot jargon or
   confabulate a plausible-sounding story that isn't actually grounded in
   the transaction's real signal.

So this module converts per-transaction SHAP values into a small, bounded
vocabulary of (semantic_label, direction, relative_strength) tuples --
e.g. ("spending pattern", "higher than usual", "3.1x normal") -- and it's
*that* structured summary, not raw feature values, that gets passed to the
LLM in summarize.py. The LLM is explicitly instructed not to invent
feature names beyond what it's given.

Backend: shap.TreeExplainer when `shap` is installed (exact, fast, works
against both XGBoost and sklearn tree ensembles). Falls back to a simple
permutation-based local attribution if shap isn't available, so the whole
pipeline still runs without the extra dependency -- less precise, but
same interface and same output shape downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .model import FEATURE_COLUMNS

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

# Maps raw engineered feature names to a human-safe semantic bucket.
# In a real deployment this mapping would come from whoever engineered the
# features (e.g. "V14 tracks velocity of spend in the last hour"). Here,
# for the synthetic dataset, we bucket by index so the mapping is stable
# and clearly not raw-feature-name leakage.
_SEMANTIC_BUCKETS = [
    "transaction timing pattern",
    "merchant category signal",
    "device / channel fingerprint",
    "spending velocity",
    "geographic consistency",
    "account tenure signal",
    "historical spend pattern",
    "billing/shipping match signal",
    "authentication strength signal",
    "network reputation signal",
    "card-present vs card-not-present signal",
    "session behavior pattern",
    "recipient/payee novelty signal",
    "cross-border activity signal",
    "time-of-day consistency",
    "purchase category diversity",
    "recent decline history",
    "device reuse across accounts",
    "IP/network consistency",
    "spend concentration signal",
    "account-age-adjusted risk signal",
    "transaction frequency signal",
    "linked-account risk signal",
    "merchant risk-tier signal",
    "browser/app fingerprint stability",
    "payment-method risk signal",
    "loyalty/history trust signal",
    "cart/checkout behavior signal",
]


def _bucket_for(feature_name: str) -> str:
    if feature_name == "Amount":
        return "transaction amount"
    idx = int(feature_name.replace("V", "")) - 1
    return _SEMANTIC_BUCKETS[idx % len(_SEMANTIC_BUCKETS)]


@dataclass
class FeatureImpact:
    semantic_label: str
    direction: str          # "elevated" | "suppressed"
    strength_label: str     # e.g. "3.1x typical influence"
    raw_shap: float          # kept for internal / audit use, never shown to LLM


class Explainer:
    def __init__(self, model_wrapper, background: pd.DataFrame):
        self.model_wrapper = model_wrapper
        self._has_shap = _HAS_SHAP
        if _HAS_SHAP:
            try:
                self._explainer = shap.TreeExplainer(model_wrapper.model)
            except Exception:
                self._has_shap = False
                self._explainer = None
        else:
            self._explainer = None
        # Used by the fallback attribution method.
        self._bg_mean = background[FEATURE_COLUMNS].mean()
        self._bg_std = background[FEATURE_COLUMNS].std().replace(0, 1.0)
        self._global_importance = self._compute_global_importance(background)

    def _compute_global_importance(self, background: pd.DataFrame) -> pd.Series:
        model = self.model_wrapper.model
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        return pd.Series(1.0, index=FEATURE_COLUMNS)

    # ---------------------------------------------------------------- shap
    def _shap_values(self, row: pd.DataFrame) -> np.ndarray:
        raw = self._explainer.shap_values(row[FEATURE_COLUMNS])
        if isinstance(raw, list):  # older shap API returns [class0, class1]
            raw = raw[1]
        return np.asarray(raw).reshape(-1)

    # ------------------------------------------------------------ fallback
    def _fallback_values(self, row: pd.DataFrame) -> np.ndarray:
        """Approximate local attribution when shap isn't installed.

        Signed z-score of each feature relative to the background
        distribution, weighted by the model's global feature importance.
        Not a substitute for true SHAP values, but gives a directionally
        sensible, per-transaction attribution with the same output shape.
        """
        z = (row[FEATURE_COLUMNS].iloc[0] - self._bg_mean) / self._bg_std
        weighted = z * self._global_importance
        return weighted.values

    # ------------------------------------------------------------- public
    def explain(self, row: pd.DataFrame, top_k: int = 3) -> List[FeatureImpact]:
        """Return the top_k most influential features for this single-row
        transaction, converted to human-safe semantic labels."""
        values = self._shap_values(row) if self._has_shap else self._fallback_values(row)
        abs_vals = np.abs(values)
        order = np.argsort(-abs_vals)

        mean_abs = abs_vals.mean() if abs_vals.mean() > 0 else 1e-6
        impacts = []
        seen_labels = set()
        for i in order:
            if len(impacts) >= top_k:
                break
            fname = FEATURE_COLUMNS[i]
            label = _bucket_for(fname)
            if label in seen_labels:
                # Skip features that collapse onto a semantic label already
                # surfaced -- a reviewer summary shouldn't cite "spending
                # velocity" twice as if they were independent signals.
                continue
            seen_labels.add(label)
            val = values[i]
            strength_ratio = abs(val) / mean_abs
            impacts.append(
                FeatureImpact(
                    semantic_label=label,
                    direction="elevated" if val > 0 else "suppressed",
                    strength_label=f"{strength_ratio:.1f}x typical influence",
                    raw_shap=float(val),
                )
            )
        return impacts

    @property
    def backend(self) -> str:
        return "shap.TreeExplainer" if self._has_shap else "fallback_zscore_attribution"
