"""
demo.py -- run the full predict -> decide -> act agent on sample
transactions and print the results a reviewer would actually see.

Run after train.py:
    python demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.model import FraudModel
from src.decision import CostBasedDecisionLayer, CostMatrix, Action
from src.explain import Explainer
from src.summarize import Summarizer
from src.pipeline import FraudAgent

MODEL_PATH = Path("models/fraud_model.pkl")
CONFIG_PATH = Path("models/decision_config.json")
DATA_PATH = Path("data/transactions.csv")

ACTION_ICON = {Action.APPROVE: "✅ APPROVE", Action.REVIEW: "🟡 REVIEW", Action.BLOCK: "🔴 BLOCK"}


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run `python train.py` first.")

    model = FraudModel.load(MODEL_PATH)
    config = json.loads(CONFIG_PATH.read_text())
    layer = CostBasedDecisionLayer(
        config["threshold_review"], config["threshold_block"], CostMatrix(**config["cost_matrix"])
    )

    df = pd.read_csv(DATA_PATH)
    background = df.sample(min(500, len(df)), random_state=1)
    explainer = Explainer(model, background)
    summarizer = Summarizer()

    print(f"LLM summary provider: {summarizer.provider}"
          + (f" ({summarizer.model})" if summarizer.model else " (no API key set, using grounded template fallback)"))
    print(f"Explanation backend: {explainer.backend}\n")

    agent = FraudAgent(model, layer, explainer, summarizer)

    # Pick a representative sample: a few fraud, a few legit, biased toward
    # transactions that will actually get flagged so the demo shows the
    # ACT layer doing something.
    fraud_rows = df[df["Class"] == 1].sample(min(4, (df["Class"] == 1).sum()), random_state=2)
    legit_rows = df[df["Class"] == 0].sample(4, random_state=3)
    sample = pd.concat([fraud_rows, legit_rows]).sample(frac=1.0, random_state=4)

    outcomes = agent.run(sample)

    for outcome in outcomes:
        print("=" * 72)
        print(f"Transaction #{outcome.index}  |  Amount: ${outcome.amount:,.2f}  |  "
              f"Fraud probability: {outcome.probability:.2%}")
        print(f"Decision: {ACTION_ICON[outcome.action]}")
        if outcome.summary:
            print(f"\nReviewer summary [{outcome.summary.provider}]:")
            print(f"  {outcome.summary.text}")
        print()


if __name__ == "__main__":
    main()
