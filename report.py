"""
report.py -- generates a self-contained HTML dashboard summarizing a
training run: model performance, the cost-sweep curve, the business-rule
cost comparison, and a table of sample decisions with reviewer summaries.

This exists because the numbers train.py prints to the console are the
whole point of the project, but they're much easier to actually absorb
as a chart than as a wall of text -- especially the cost-sweep curve
(how flat it is) and the business-rule comparison (how much worse the
"obvious" rule is).

Run after train.py:
    python report.py
Output:
    outputs/report.html         (self-contained, open in any browser)
    outputs/roc_curve.png
    outputs/cost_sweep.png
    outputs/business_rule.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.model import FraudModel
from src.decision import CostBasedDecisionLayer, CostMatrix, evaluate_business_rule, Action
from src.explain import Explainer
from src.summarize import Summarizer
from src.pipeline import FraudAgent

MODEL_PATH = Path("models/fraud_model.pkl")
CONFIG_PATH = Path("models/decision_config.json")
TEST_SPLIT_PATH = Path("models/test_split.csv")
DATA_PATH = Path("data/transactions.csv")
OUT_DIR = Path("outputs")

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.8,
    "font.size": 11,
}


def _save_roc_curve(y_test, proba_test, path: Path):
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    roc_auc = auc(fpr, tpr)
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#1f6feb", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("PREDICT layer: ROC curve (held-out test set)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def _save_cost_sweep(sweep_df: pd.DataFrame, best_threshold: float, path: Path):
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sweep_df["threshold"], sweep_df["total_cost"], color="#1f6feb", linewidth=2)
        ax.axvline(best_threshold, color="#da3633", linestyle="--", linewidth=1.2,
                   label=f"best single threshold ({best_threshold:.2f})")
        ax.set_xlabel("Single approve/block threshold")
        ax.set_ylabel("Total cost ($)")
        ax.set_title("DECIDE layer: cost vs. threshold\n(flat curve = threshold isn't the real lever)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def _save_business_rule_chart(base_cost: float, rule_cost: float, path: Path):
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        bars = ax.bar(
            ["Tuned cost-based\ndecision layer", "+ blanket\n\"flag >$500\" rule"],
            [base_cost, rule_cost],
            color=["#2ea043", "#da3633"],
            width=0.55,
        )
        for bar, val in zip(bars, [base_cost, rule_cost]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"${val:,.0f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("Total cost ($)")
        ax.set_title("An 'obviously correct' rule\ncan make total cost worse, not better")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def _decision_rows_html(agent: FraudAgent, sample: pd.DataFrame) -> str:
    icon = {Action.APPROVE: ("✅", "#2ea043"), Action.REVIEW: ("🟡", "#9a6700"), Action.BLOCK: ("🔴", "#da3633")}
    outcomes = agent.run(sample)
    rows = []
    for o in outcomes:
        emoji, color = icon[o.action]
        summary_html = o.summary.text if o.summary else "<em>(auto-approved, no review needed)</em>"
        rows.append(f"""
        <tr>
          <td>{o.index}</td>
          <td>${o.amount:,.2f}</td>
          <td>{o.probability:.1%}</td>
          <td><span style="color:{color}; font-weight:600;">{emoji} {o.action.value.upper()}</span></td>
          <td>{summary_html}</td>
        </tr>""")
    return "\n".join(rows)


def build_report():
    if not TEST_SPLIT_PATH.exists():
        raise SystemExit("No saved test split found. Run `python train.py` first (this version of it, "
                          "which now saves models/test_split.csv).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = FraudModel.load(MODEL_PATH)
    config = json.loads(CONFIG_PATH.read_text())
    df = pd.read_csv(DATA_PATH)

    test_split = pd.read_csv(TEST_SPLIT_PATH, index_col="orig_index")
    proba_test = test_split["fraud_probability"].values
    y_test = test_split["Class"]
    amounts_test = test_split["Amount"].values

    layer = CostBasedDecisionLayer(
        config["threshold_review"], config["threshold_block"], CostMatrix(**config["cost_matrix"])
    )

    sweep = layer.sweep_single_threshold(y_test.values, proba_test)
    best_row = sweep.loc[sweep["total_cost"].idxmin()]
    rule_result = evaluate_business_rule(y_test.values, proba_test, amounts_test, layer, amount_threshold=500.0)

    _save_roc_curve(y_test, proba_test, OUT_DIR / "roc_curve.png")
    _save_cost_sweep(sweep, best_row["threshold"], OUT_DIR / "cost_sweep.png")
    _save_business_rule_chart(rule_result["base_cost"], rule_result["rule_cost"], OUT_DIR / "business_rule.png")

    background = df.sample(min(500, len(df)), random_state=1)
    explainer = Explainer(model, background)
    summarizer = Summarizer()
    agent = FraudAgent(model, layer, explainer, summarizer)

    fraud_rows = df[df["Class"] == 1].sample(min(3, (df["Class"] == 1).sum()), random_state=2)
    legit_rows = df[df["Class"] == 0].sample(3, random_state=3)
    sample = pd.concat([fraud_rows, legit_rows]).sample(frac=1.0, random_state=4)
    decision_rows_html = _decision_rows_html(agent, sample)

    cost_flat = (sweep["total_cost"].max() - sweep["total_cost"].min()) < 0.05 * sweep["total_cost"].max()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Agentic Fraud Detection -- Run Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          max-width: 980px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.55; }}
  h1 {{ font-size: 1.7em; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-top: 0; margin-bottom: 32px; }}
  h2 {{ margin-top: 44px; border-bottom: 2px solid #eee; padding-bottom: 6px; }}
  .grid {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .card {{ background: #f8f9fa; border: 1px solid #e5e5e5; border-radius: 10px; padding: 18px 20px; flex: 1; min-width: 220px; }}
  .card .label {{ color: #666; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.04em; }}
  .card .value {{ font-size: 1.6em; font-weight: 700; margin-top: 4px; }}
  img {{ max-width: 100%; border-radius: 8px; border: 1px solid #e5e5e5; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.92em; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #eee; vertical-align: top; }}
  th {{ background: #f8f9fa; }}
  .callout {{ background: #fff8e6; border-left: 4px solid #d4a72c; padding: 12px 16px; border-radius: 4px; margin: 16px 0; }}
  .good {{ border-left-color: #2ea043; background: #e9f7ee; }}
  .bad {{ border-left-color: #da3633; background: #fdeeee; }}
  .footer {{ margin-top: 50px; color: #888; font-size: 0.85em; border-top: 1px solid #eee; padding-top: 16px; }}
</style>
</head>
<body>

<h1>Agentic Fraud Detection -- Run Report</h1>
<p class="subtitle">predict &rarr; decide &rarr; act, generated by report.py</p>

<div class="grid">
  <div class="card"><div class="label">Backend</div><div class="value">{config['backend']}</div></div>
  <div class="card"><div class="label">ROC-AUC</div><div class="value">{config['roc_auc']:.4f}</div></div>
  <div class="card"><div class="label">PR-AUC</div><div class="value">{config['pr_auc']:.4f}</div></div>
  <div class="card"><div class="label">Explanation backend</div><div class="value" style="font-size:1.1em;">{explainer.backend}</div></div>
</div>

<h2>Predict: model performance</h2>
<img src="roc_curve.png" alt="ROC curve">

<h2>Decide: is the threshold actually the lever?</h2>
<img src="cost_sweep.png" alt="Cost sweep curve">
<div class="callout {'bad' if cost_flat else 'good'}">
{"The cost curve is nearly flat across the whole threshold range &mdash; moving the cutoff barely changes total cost. That points at a small number of confidently-wrong predictions as the real cost driver, not threshold placement." if cost_flat else "The cost curve has a clear minimum &mdash; threshold placement meaningfully affects total cost here."}
</div>

<h2>Decide: the "obviously correct" rule check</h2>
<img src="business_rule.png" alt="Business rule comparison">
<div class="callout bad">
Stacking a blanket "flag every transaction over $500" rule on top of the tuned decision layer
changed total cost by <strong>{rule_result['multiplier']:.2f}x</strong>
(${rule_result['base_cost']:,.0f} &rarr; ${rule_result['rule_cost']:,.0f}).
{"This is worse, not better -- the rule floods reviewers with legitimate large purchases, and that cost outweighs the fraud it incidentally catches. Not applying it." if rule_result['multiplier'] > 1 else "In this run the rule happened to reduce cost -- re-run train.py with a different seed to see how much this varies."}
</div>

<h2>Act: sample decisions with reviewer summaries</h2>
<p style="color:#666; margin-top:-6px;">Summaries generated by: <strong>{summarizer.provider}</strong>{f" ({summarizer.model})" if summarizer.model else " -- no API key set, using the grounded template fallback"}</p>
<table>
  <thead>
    <tr><th>Txn</th><th>Amount</th><th>Fraud prob.</th><th>Decision</th><th>Reviewer summary</th></tr>
  </thead>
  <tbody>
    {decision_rows_html}
  </tbody>
</table>

<div class="footer">
  Generated by <code>report.py</code> from <code>models/fraud_model.pkl</code> and
  <code>models/decision_config.json</code>. Re-run <code>python train.py</code> then
  <code>python report.py</code> to refresh with a new random seed or dataset.
</div>

</body>
</html>
"""
    (OUT_DIR / "report.html").write_text(html)
    print(f"Report written to {OUT_DIR / 'report.html'}")
    print(f"Charts written to {OUT_DIR}/*.png")


if __name__ == "__main__":
    build_report()
