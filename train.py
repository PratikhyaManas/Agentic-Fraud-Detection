"""
train.py -- end-to-end training + cost-analysis entry point.

Run:
    python train.py

What it does, in order:
  1. Generates the synthetic transaction dataset (if not already present).
  2. Trains the PREDICT model (XGBoost, or sklearn fallback) and reports
     ROC-AUC / PR-AUC on a held-out split.
  3. Runs a single-threshold cost sweep and reports how flat/sharp the
     cost curve is -- this is the check that revealed, in the reference
     project, that threshold tuning alone barely moved total cost.
  4. Fits the two-threshold (review/block) cost-based decision layer.
  5. Runs the "always flag transactions over $500" business-rule check
     against the tuned baseline, and reports the cost multiplier -- this
     is the check that revealed the rule made things ~5x worse.
  6. Saves the trained model + chosen thresholds to models/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.generate_data import generate
from src.model import FraudModel
from src.decision import CostBasedDecisionLayer, CostMatrix, evaluate_business_rule
from src.runtime import maybe_load_dotenv

DATA_PATH = Path("data/transactions.csv")
MODEL_PATH = Path("models/fraud_model.pkl")
CONFIG_PATH = Path("models/decision_config.json")
TEST_SPLIT_PATH = Path("models/test_split.csv")


def main():
    maybe_load_dotenv()

    if not DATA_PATH.exists():
        print(f"Generating synthetic dataset -> {DATA_PATH}")
        df = generate(n_legit=10_000, n_fraud=200, n_confidently_missed=3)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df):,} transactions ({df['Class'].sum()} fraud, "
          f"{df['Class'].mean():.3%} fraud rate)")

    model = FraudModel()
    result = model.fit(df)
    print(f"\nBackend: {result.backend}")
    print(f"ROC-AUC: {result.roc_auc:.4f}   PR-AUC: {result.pr_auc:.4f}")
    print(f"Test set: {result.n_test:,} transactions, {result.n_fraud_test} fraud")

    y_test, proba_test, X_test = model._y_test, model._proba_test, model._X_test
    amounts_test = X_test["Amount"].values

    cost_matrix = CostMatrix(cost_false_negative=500.0, cost_false_positive=25.0, cost_review=3.0)
    layer = CostBasedDecisionLayer(0.5, 0.5, cost_matrix)

    print("\n--- Single-threshold cost sweep (review==block) ---")
    sweep = layer.sweep_single_threshold(y_test.values, proba_test)
    best_row = sweep.loc[sweep["total_cost"].idxmin()]
    worst_row = sweep.loc[sweep["total_cost"].idxmax()]
    spread = worst_row["total_cost"] - best_row["total_cost"]
    print(f"Best single threshold: {best_row['threshold']:.2f} -> cost ${best_row['total_cost']:.2f}")
    print(f"Cost range across ALL thresholds tested: ${spread:.2f} "
          f"(best ${best_row['total_cost']:.2f} vs worst ${worst_row['total_cost']:.2f})")
    if spread < 0.05 * worst_row["total_cost"]:
        print("-> Cost curve is nearly flat: threshold placement barely matters here.")
        print("   That's a signal the real cost driver is confidently-wrong predictions,")
        print("   not where the cutoff sits. See the missed-fraud audit below.")

    print("\n--- Fitting two-threshold (review/block) cost-based layer ---")
    layer.fit_thresholds(y_test.values, proba_test)
    tuned_cost = layer.total_cost(y_test.values, proba_test)
    print(f"Chosen thresholds: review >= {layer.threshold_review:.3f}, block >= {layer.threshold_block:.3f}")
    print(f"Total cost at tuned thresholds: ${tuned_cost:.2f}")

    print("\n--- Business-rule check: 'always flag transactions over $500' ---")
    rule_result = evaluate_business_rule(y_test.values, proba_test, amounts_test, layer, amount_threshold=500.0)
    print(f"Baseline (tuned) cost: ${rule_result['base_cost']:.2f}")
    print(f"With the >$500 rule stacked on top: ${rule_result['rule_cost']:.2f}")
    print(f"Multiplier: {rule_result['multiplier']:.2f}x", end="")
    if rule_result["multiplier"] > 1.5:
        print("  <-- the 'obviously correct' rule makes things worse. Not applying it.")
    else:
        print()

    print("\n--- Confidently-wrong audit (fraud missed with near-zero probability) ---")
    y_test_arr = y_test.values
    fraud_mask = y_test_arr == 1
    fraud_probs = proba_test[fraud_mask]
    fraud_amounts = amounts_test[fraud_mask]
    low_conf_mask = fraud_probs < 0.01
    n_confidently_missed = int(low_conf_mask.sum())
    print(f"Fraud cases in test set scored below 1% probability: {n_confidently_missed} "
          f"out of {int(fraud_mask.sum())} total fraud cases")
    if n_confidently_missed:
        missed_value = fraud_amounts[low_conf_mask].sum()
        print(f"Total value of confidently-missed fraud: ${missed_value:,.2f}")
        print("These are not 'almost caught' -- no threshold or rule fixes them;")
        print("they need better features/signal, not better decisioning.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "threshold_review": layer.threshold_review,
            "threshold_block": layer.threshold_block,
            "cost_matrix": vars(cost_matrix),
            "backend": result.backend,
            "roc_auc": result.roc_auc,
            "pr_auc": result.pr_auc,
        }, f, indent=2)

    # Persist the held-out test split + predictions so report.py (and any
    # other downstream tool) can reproduce the exact charts/metrics above
    # without needing to retrain or guess the random split.
    test_split = X_test.copy()
    test_split["Class"] = y_test.values
    test_split["fraud_probability"] = proba_test
    test_split.to_csv(TEST_SPLIT_PATH, index=True, index_label="orig_index")

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved decision config -> {CONFIG_PATH}")
    print(f"Saved test split + predictions -> {TEST_SPLIT_PATH}")


if __name__ == "__main__":
    main()
