from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .decision import CostBasedDecisionLayer, CostMatrix


@dataclass
class SchemaReport:
    required_columns: list[str]
    missing_columns: list[str]
    non_numeric_required_columns: list[str]
    row_count: int

    @property
    def ok(self) -> bool:
        return not self.missing_columns and not self.non_numeric_required_columns

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "required_columns": self.required_columns,
            "missing_columns": self.missing_columns,
            "non_numeric_required_columns": self.non_numeric_required_columns,
            "row_count": self.row_count,
        }


def build_schema_report(df: pd.DataFrame, required_columns: list[str]) -> SchemaReport:
    missing = [c for c in required_columns if c not in df.columns]
    non_numeric = []
    for col in required_columns:
        if col not in df.columns:
            continue
        # Treat columns as invalid when values cannot be parsed as numeric.
        parsed = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = parsed.isna() & df[col].notna()
        if invalid_mask.any():
            non_numeric.append(col)
    return SchemaReport(
        required_columns=required_columns,
        missing_columns=missing,
        non_numeric_required_columns=non_numeric,
        row_count=int(len(df)),
    )


def evaluate_threshold_sensitivity(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    named_cost_matrices: list[dict],
    threshold_grid: Optional[np.ndarray] = None,
) -> dict:
    threshold_grid = threshold_grid if threshold_grid is not None else np.linspace(0.01, 0.99, 99)
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    scenarios = []
    for scenario in named_cost_matrices:
        name = scenario["name"]
        cm_kwargs = scenario["cost_matrix"]
        cm = CostMatrix(**cm_kwargs)

        layer = CostBasedDecisionLayer(0.5, 0.9, cm)
        sweep_df = layer.sweep_single_threshold(y_true, probabilities, grid=threshold_grid)
        best_row = sweep_df.loc[sweep_df["total_cost"].idxmin()]
        worst_row = sweep_df.loc[sweep_df["total_cost"].idxmax()]

        tuned_layer = CostBasedDecisionLayer(0.5, 0.9, cm)
        tuned_layer.fit_thresholds(y_true, probabilities)
        tuned_cost = tuned_layer.total_cost(y_true, probabilities)

        scenarios.append(
            {
                "name": name,
                "cost_matrix": cm_kwargs,
                "single_threshold": {
                    "best_threshold": float(best_row["threshold"]),
                    "best_total_cost": float(best_row["total_cost"]),
                    "worst_total_cost": float(worst_row["total_cost"]),
                    "spread": float(worst_row["total_cost"] - best_row["total_cost"]),
                },
                "two_threshold": {
                    "review_threshold": float(tuned_layer.threshold_review),
                    "block_threshold": float(tuned_layer.threshold_block),
                    "total_cost": float(tuned_cost),
                },
            }
        )

    return {
        "n_rows": int(len(y_true)),
        "n_scenarios": len(scenarios),
        "scenarios": scenarios,
    }