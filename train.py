#!/usr/bin/env python3
"""
Train the fraud model on synthetic data and persist artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_generator import generate_transactions, train_test_split_df, FEATURE_NAMES
from src.model import FraudModel

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("Generating synthetic transactions...")
    df = generate_transactions(n_samples=40_000, fraud_rate=0.018, random_state=42)
    train_df, test_df = train_test_split_df(df, test_size=0.2)

    train_df.to_parquet(DATA_DIR / "train.parquet", index=False)
    test_df.to_parquet(DATA_DIR / "test.parquet", index=False)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print(f"  Fraud rate (train): {train_df['is_fraud'].mean():.3%}")

    print("\nTraining XGBoost model...")
    model = FraudModel(n_estimators=250, max_depth=5, learning_rate=0.07)
    model.fit(train_df[FEATURE_NAMES], train_df["is_fraud"])

    print("\nEvaluating on hold-out test set...")
    metrics = model.evaluate(test_df[FEATURE_NAMES], test_df["is_fraud"])
    print(f"  ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC  : {metrics['pr_auc']:.4f}")
    print("\nClassification report (threshold 0.5):")
    report = metrics["report"]
    for label in ("0", "1"):
        r = report[label]
        print(
            f"  class {label}: precision={r['precision']:.3f}  "
            f"recall={r['recall']:.3f}  f1={r['f1-score']:.3f}"
        )

    model.save(MODEL_DIR)
    # Also dump metrics for the README / demo
    serialisable = {
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "report": {
            k: {sk: float(sv) for sk, sv in v.items()} if isinstance(v, dict) else v
            for k, v in report.items()
        },
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(serialisable, indent=2))
    print(f"\nModel + metrics saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()