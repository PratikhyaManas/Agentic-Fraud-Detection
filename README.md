# Agentic Fraud-Detection System

A from-scratch implementation of the **Predict → Decide → Act** architecture described in the Medium article [*What Building an Agentic Fraud-Detection System Taught Me About Agentic AI*](https://medium.com/@Abd24205/what-building-an-agentic-fraud-detection-system-taught-me-about-agentic-ai-a1e151257ce0).

This is **not** a traditional score-and-threshold pipeline. The system carries every transaction all the way to an actionable outcome and (when needed) a human-readable explanation.

---

## Architecture

| Layer | Responsibility | Implementation |
|-------|----------------|----------------|
| **Predict** | Output P(fraud) | XGBoost classifier + SHAP TreeExplainer |
| **Decide** | Map probability → action using real business costs | Cost-sensitive expected-cost minimiser (APPROVE / FLAG / BLOCK) |
| **Act** | Produce an analyst-ready summary | Grounded natural-language generator (LLM-ready interface; template engine used offline) |

```
Transaction
    │
    ▼
┌─────────────┐
│   Predict   │  XGBoost → p_fraud + SHAP impacts
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Decide    │  minimise E[cost] under FP/FN/review costs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Act      │  reviewer summary (only for FLAG / BLOCK)
└─────────────┘
```

### Why cost-based decisioning matters

A fixed probability threshold ignores the asymmetric cost of mistakes. In this domain a false negative (missed fraud) is typically 20–50× more expensive than a false positive. The decision layer therefore:

1. Computes the *expected cost* of each action given `p_fraud` and transaction amount.
2. Applies soft probability guardrails (`flag_threshold`, `block_threshold`).
3. Selects the minimum-cost action.

You can change the cost matrix in `src/decision.py` (`CostConfig`) without retraining the model.

### Grounded explanations

The Act layer never sees raw internal feature names that would be meaningless to an analyst. It only receives:

- direction of SHAP impact (`increases_fraud_risk` / `decreases_fraud_risk`)
- human-readable labels
- the actual feature value

This mirrors the article’s key lesson: keep the LLM (or template engine) strictly grounded so it cannot invent plausible-sounding but incorrect reasons.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train on synthetic data (~40 k transactions)
python train.py

# 3. Run the live demo (prints full traces)
python demo.py
```

Example output fragment:

```
--- Transaction TXN-00012345 ---
  Amount          : $1,842.00
  Ground-truth    : FRAUD
  P(fraud)        : 91.4%
  Action          : BLOCK
  Decision reason : Probability 0.914 exceeds hard block threshold (0.85)...
  Reviewer summary:
    A $1,842.00 payment was blocked because the fraud probability reached 91.4%.
    The primary driver is distance from the previous transaction (842 km), which
    strongly elevates risk. Secondary signal: online order = yes.
    Hard block applied; customer will need to contact support to proceed.
```

---

## Project layout

```
agentic_fraud_detection/
├── train.py                 # generate data + train + persist model
├── demo.py                  # end-to-end Predict→Decide→Act traces
├── requirements.txt
├── README.md
├── data/                    # train/test parquet (created by train.py)
├── models/                  # xgb_model.joblib + metrics.json
└── src/
    ├── data_generator.py    # synthetic transaction generator
    ├── model.py             # FraudModel (XGBoost + SHAP)
    ├── decision.py          # CostBasedDecisionMaker
    ├── explainer_agent.py   # ReviewerSummaryAgent (Act layer)
    └── agent.py             # FraudDetectionAgent orchestrator
```

---

## Using the agent in your own code

```python
from src.agent import FraudDetectionAgent
import pandas as pd

agent = FraudDetectionAgent.from_pretrained("models")

# single transaction (must contain the feature columns)
row = pd.Series({
    "transaction_id": "TXN-demo-001",
    "amount": 1250.0,
    "hour_of_day": 2,
    "day_of_week": 5,
    "distance_from_home_km": 420.0,
    "distance_from_last_txn_km": 380.0,
    "ratio_to_median_purchase_price": 4.7,
    "repeat_retailer": 0,
    "used_chip": 0,
    "used_pin_number": 0,
    "online_order": 1,
})

result = agent.process(row)
print(result.action)            # e.g. "FLAG"
print(result.reviewer_summary)
print(result.top_shap_impacts)
```

---

## Swapping in a real LLM

`src/explainer_agent.py` already contains a commented `LLMReviewerSummaryAgent` skeleton that talks to the OpenAI-compatible Groq (or any other) endpoint. The prompt contract is identical: feed only SHAP *directions* and human labels, never raw model internals. Replace the default `ReviewerSummaryAgent` when you have network access and an API key.

---

## Design notes / lessons mirrored from the article

1. **Threshold sweeps alone are often useless** when FN cost ≫ FP cost — the decision boundary barely moves expected cost. The interesting failures are the *confidently wrong* cases the model never approached.
2. **Naïve business rules hurt**. An “always flag > $500” rule flooded reviewers with false positives and raised total cost dramatically.
3. **Repetition is a bigger practical problem than hallucination** for reviewer summaries. Variation templates (or a single carefully chosen one-shot example) help more than “please be creative”.
4. **Grounding is non-negotiable**. Withholding feature names that are meaningless to a human (or that the model could misuse) is the main defence against plausible-but-wrong explanations.

---

## Licence

MIT — use freely for learning, demos, or as a starting point for production systems.
