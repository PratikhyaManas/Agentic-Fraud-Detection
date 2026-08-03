import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import build_schema_report, evaluate_threshold_sensitivity


def test_schema_report_detects_missing_and_non_numeric_columns():
    df = pd.DataFrame(
        {
            "V1": [0.1, 0.2],
            "V2": ["bad", "0.5"],
            "Amount": [12.0, 20.0],
        }
    )
    required = ["V1", "V2", "V3", "Amount"]
    report = build_schema_report(df, required)

    assert report.ok is False
    assert report.missing_columns == ["V3"]
    assert "V2" in report.non_numeric_required_columns


def test_threshold_sensitivity_returns_expected_shape():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    scenarios = [
        {
            "name": "default",
            "cost_matrix": {
                "cost_false_negative": 500.0,
                "cost_false_positive": 25.0,
                "cost_review": 3.0,
                "cost_true_positive_extra": 0.0,
            },
        },
        {
            "name": "high_review_cost",
            "cost_matrix": {
                "cost_false_negative": 500.0,
                "cost_false_positive": 25.0,
                "cost_review": 10.0,
                "cost_true_positive_extra": 0.0,
            },
        },
    ]

    out = evaluate_threshold_sensitivity(y_true, proba, scenarios)
    assert out["n_rows"] == 6
    assert out["n_scenarios"] == 2
    assert len(out["scenarios"]) == 2
    assert out["scenarios"][0]["name"] == "default"
    assert "single_threshold" in out["scenarios"][0]
    assert "two_threshold" in out["scenarios"][0]