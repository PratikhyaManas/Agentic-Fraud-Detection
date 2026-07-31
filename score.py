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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.model import FraudModel, FEATURE_COLUMNS
from src.decision import CostBasedDecisionLayer, CostMatrix
from src.explain import Explainer
from src.summarize import Summarizer
from src.pipeline import FraudAgent

MODEL_PATH = Path("models/fraud_model.pkl")
CONFIG_PATH = Path("models/decision_config.json")


def main():
    parser = argparse.ArgumentParser(description="Batch-score transactions through the fraud agent")
    parser.add_argument("--input", required=True, help="Path to a transactions CSV (needs V1..V28, Amount columns)")
    parser.add_argument("--output", required=True, help="Where to write the scored CSV")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N rows (for quick tests)")
    parser.add_argument("--background", default="data/transactions.csv",
                        help="CSV used as the background distribution for explanations")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run `python train.py` first.")

    model = FraudModel.load(MODEL_PATH)
    config = json.loads(CONFIG_PATH.read_text())
    layer = CostBasedDecisionLayer(
        config["threshold_review"], config["threshold_block"], CostMatrix(**config["cost_matrix"])
    )

    df = pd.read_csv(args.input)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Input file is missing required columns: {missing}")
    if args.limit:
        df = df.head(args.limit)

    background = pd.read_csv(args.background)
    explainer = Explainer(model, background.sample(min(500, len(background)), random_state=1))
    summarizer = Summarizer()
    agent = FraudAgent(model, layer, explainer, summarizer)

    print(f"Scoring {len(df):,} transactions "
          f"(summaries via: {summarizer.provider}{' -- ' + summarizer.model if summarizer.model else ''})...")
    outcomes = agent.run(df)

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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nWrote {len(results):,} scored rows -> {out_path}")


if __name__ == "__main__":
    main()
