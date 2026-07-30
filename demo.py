#!/usr/bin/env python3
"""
End-to-end demo of the agentic fraud-detection system.

Loads a trained model (or trains one on the fly if missing), samples a few
transactions from the test set, and prints the full Predict → Decide → Act
trace for each.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agent import FraudDetectionAgent
from src.data_generator import FEATURE_NAMES, generate_transactions, train_test_split_df
from src.model import FraudModel

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


def ensure_model() -> FraudDetectionAgent:
    if (MODEL_DIR / "xgb_model.joblib").exists():
        print("Loading pre-trained model...")
        return FraudDetectionAgent.from_pretrained(MODEL_DIR)

    print("No saved model found — training a quick one...")
    df = generate_transactions(n_samples=15_000, random_state=7)
    train_df, _ = train_test_split_df(df)
    model = FraudModel(n_estimators=120, max_depth=4)
    model.fit(train_df[FEATURE_NAMES], train_df["is_fraud"])
    model.save(MODEL_DIR)
    return FraudDetectionAgent(model=model)


def load_sample_transactions(n: int = 8) -> pd.DataFrame:
    test_path = DATA_DIR / "test.parquet"
    if test_path.exists():
        df = pd.read_parquet(test_path)
    else:
        df = generate_transactions(n_samples=5_000, random_state=99)
    # Prefer a mix of high- and low-risk looking rows
    fraud = df[df["is_fraud"] == 1].sample(min(n // 2, len(df[df["is_fraud"] == 1])), random_state=1)
    legit = df[df["is_fraud"] == 0].sample(n - len(fraud), random_state=1)
    return pd.concat([fraud, legit]).sample(frac=1, random_state=2).reset_index(drop=True)


def main() -> None:
    agent = ensure_model()
    samples = load_sample_transactions(8)

    print("\n" + "=" * 72)
    print("AGENTIC FRAUD DETECTION — LIVE TRACE")
    print("=" * 72)

    results = []
    for idx, row in samples.iterrows():
        out = agent.process(row)
        results.append(out.to_dict())

        print(f"\n--- Transaction {out.transaction_id} ---")
        print(f"  Amount          : ${out.amount:,.2f}")
        print(f"  Ground-truth    : {'FRAUD' if row['is_fraud'] == 1 else 'legit'}")
        print(f"  P(fraud)        : {out.fraud_probability:.3%}")
        print(f"  Action          : {out.action}")
        print(f"  Decision reason : {out.decision_rationale}")
        print(f"  Expected costs  : approve={out.expected_costs['approve']:.1f}  "
              f"flag={out.expected_costs['flag']:.1f}  "
              f"block={out.expected_costs['block']:.1f}")
        print(f"  Reviewer summary:\n    {out.reviewer_summary}")
        print("  Top SHAP impacts:")
        for imp in out.top_shap_impacts[:3]:
            print(
                f"    • {imp['feature']:<32} "
                f"value={imp['value']:<10.3g}  "
                f"shap={imp['shap']:+.4f}  ({imp['direction']})"
            )

    # Persist a machine-readable run log
    out_path = ROOT / "demo_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()