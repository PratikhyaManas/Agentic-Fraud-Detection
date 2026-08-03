"""
score.py -- batch-score a transactions CSV through the full
predict -> decide -> act pipeline and write the results out.

This is the "use it on your own data" entry point, as opposed to
train.py (builds the model) and demo.py (prints a few examples to the
console). Any CSV with the same columns as the synthetic dataset
(V1..V28, Amount -- Class is optional/ignored) works.

Usage:
    python score.py --input data/transactions.csv --output outputs/scored.csv
    python score.py --input my_transactions.csv --output results.csv --limit 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.decision import Action
from src.evaluation import build_schema_report, evaluate_threshold_sensitivity
from src.model import FEATURE_COLUMNS
from src.summarize import Summarizer
from src.runtime import (
    build_agent,
    load_model_and_decision_layer,
    maybe_load_dotenv,
)

MODEL_PATH = Path("models/fraud_model.pkl")
CONFIG_PATH = Path("models/decision_config.json")


def main():
    maybe_load_dotenv()

    parser = argparse.ArgumentParser(description="Batch-score transactions through the fraud agent")
    parser.add_argument("--input", required=True, help="Path to a transactions CSV (needs V1..V28, Amount columns)")
    parser.add_argument("--output", required=True, help="Where to write the scored CSV")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N rows (for quick tests)")
    parser.add_argument("--background", default="data/transactions.csv",
                        help="CSV used as the background distribution for explanations")
    parser.add_argument("--top-k-features", type=int, default=3,
                        help="Top semantic signals to include per explained transaction")
    parser.add_argument("--no-summaries", action="store_true",
                        help="Skip explanation + summary generation for faster pure decision scoring")
    parser.add_argument("--metrics-output", default=None,
                        help="Optional path to write evaluation metrics JSON (only when Class column exists)")
    parser.add_argument("--schema-report-output", default=None,
                        help="Optional path to write schema validation report JSON")
    parser.add_argument("--strict-schema", action="store_true",
                        help="Fail if required feature columns are non-numeric")
    parser.add_argument("--cost-sensitivity-output", default=None,
                        help="Optional path to write threshold/cost sensitivity JSON (requires Class column)")
    parser.add_argument("--cost-matrices", default=None,
                        help="Optional JSON file with named cost matrices for sensitivity analysis")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run `python train.py` first.")

    model, layer, config = load_model_and_decision_layer(MODEL_PATH, CONFIG_PATH)

    df = pd.read_csv(args.input)
    schema_report = build_schema_report(df, FEATURE_COLUMNS)
    if schema_report.missing_columns:
        raise SystemExit(f"Input file is missing required columns: {schema_report.missing_columns}")
    if args.strict_schema and schema_report.non_numeric_required_columns:
        raise SystemExit(
            "Input file has non-numeric required columns: "
            f"{schema_report.non_numeric_required_columns}"
        )
    if schema_report.non_numeric_required_columns:
        print(
            "Warning: non-numeric values detected in required columns "
            f"{schema_report.non_numeric_required_columns}; model scoring may fail. "
            "Use --strict-schema to fail fast."
        )
    if args.limit:
        df = df.head(args.limit)

    background = None if args.no_summaries else pd.read_csv(args.background)
    summarizer = None if args.no_summaries else Summarizer()
    explain_actions = () if args.no_summaries else (Action.REVIEW, Action.BLOCK)
    agent = build_agent(
        model=model,
        decision_layer=layer,
        background=background,
        summarizer=summarizer,
        explain_actions=explain_actions,
    )

    if summarizer is None:
        provider_desc = "disabled"
    else:
        provider_desc = f"{summarizer.provider}{' -- ' + summarizer.model if summarizer.model else ''}"
    print(f"Scoring {len(df):,} transactions (summaries: {provider_desc})...")
    outcomes = agent.run(df, top_k_features=args.top_k_features)

    results = pd.DataFrame([
        {
            "index": o.index,
            "amount": o.amount,
            "fraud_probability": o.probability,
            "action": o.action.value,
            "top_signals": "; ".join(f"{i.semantic_label} ({i.direction})" for i in o.impacts) if o.impacts else "",
            "reviewer_summary": o.summary.text if o.summary else "",
        }
        for o in outcomes
    ])

    counts = results["action"].value_counts()
    print("\nAction breakdown:")
    for action, n in counts.items():
        print(f"  {action:>8}: {n:,}")

    metrics = None
    if "Class" in df.columns:
        y_true = df["Class"].astype(int).values
        proba = results["fraud_probability"].values
        action = results["action"]
        metrics = {
            "n_rows": int(len(df)),
            "fraud_rate": float(df["Class"].mean()),
            "roc_auc": float(roc_auc_score(y_true, proba)),
            "pr_auc": float(average_precision_score(y_true, proba)),
            "decision_cost_total": float(layer.total_cost(y_true, proba)),
            "decision_thresholds": {
                "review": float(layer.threshold_review),
                "block": float(layer.threshold_block),
            },
            "action_counts": {k: int(v) for k, v in counts.to_dict().items()},
            "action_outcomes": {
                "approved_fraud": int(((action == "approve") & (df["Class"] == 1)).sum()),
                "blocked_legit": int(((action == "block") & (df["Class"] == 0)).sum()),
                "reviewed_total": int((action == "review").sum()),
            },
        }
        print("\nEvaluation (ground truth Class found):")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")
        print(f"  Total decision cost: ${metrics['decision_cost_total']:.2f}")

    if args.schema_report_output:
        schema_path = Path(args.schema_report_output)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(json.dumps(schema_report.to_dict(), indent=2))
        print(f"Wrote schema report JSON -> {schema_path}")

    if args.metrics_output:
        if metrics is None:
            raise SystemExit("--metrics-output requires a Class column in the input CSV.")
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"Wrote metrics JSON -> {metrics_path}")

    if args.cost_sensitivity_output:
        if "Class" not in df.columns:
            raise SystemExit("--cost-sensitivity-output requires a Class column in the input CSV.")
        y_true = df["Class"].astype(int).values
        proba = results["fraud_probability"].values

        if args.cost_matrices:
            named_cost_matrices = json.loads(Path(args.cost_matrices).read_text())
        else:
            named_cost_matrices = [
                {
                    "name": "trained_default",
                    "cost_matrix": config["cost_matrix"],
                }
            ]
        sensitivity = evaluate_threshold_sensitivity(y_true, proba, named_cost_matrices)
        sensitivity_path = Path(args.cost_sensitivity_output)
        sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
        sensitivity_path.write_text(json.dumps(sensitivity, indent=2))
        print(f"Wrote cost sensitivity JSON -> {sensitivity_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nWrote {len(results):,} scored rows -> {out_path}")


if __name__ == "__main__":
    main()
