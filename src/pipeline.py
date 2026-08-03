"""
pipeline.py -- wires PREDICT -> DECIDE -> ACT into a single agentic pipeline.

This is the piece that turns three separate components into "an agent":
given a raw transaction, it produces not just a score but a decision, and
for anything non-trivial (review or block), a grounded, human-readable
explanation. Everything upstream (model.py, decision.py, explain.py,
summarize.py) is deliberately usable standalone -- this module is just the
thin composition layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .model import FraudModel, FEATURE_COLUMNS
from .decision import CostBasedDecisionLayer, Action
from .explain import Explainer, FeatureImpact
from .summarize import Summarizer, Summary


@dataclass
class TransactionOutcome:
    index: int
    amount: float
    probability: float
    action: Action
    impacts: List[FeatureImpact] = field(default_factory=list)
    summary: Optional[Summary] = None


class FraudAgent:
    """The full predict -> decide -> act pipeline for one or many
    transactions."""

    def __init__(
        self,
        model: FraudModel,
        decision_layer: CostBasedDecisionLayer,
        explainer: Optional[Explainer] = None,
        summarizer: Optional[Summarizer] = None,
        explain_actions=(Action.REVIEW, Action.BLOCK),
    ):
        self.model = model
        self.decision_layer = decision_layer
        self.explainer = explainer
        self.explain_actions = set(explain_actions)
        if self.explain_actions and self.explainer is None:
            raise ValueError("An explainer is required when explain_actions is non-empty.")
        self.summarizer = summarizer or (Summarizer() if self.explain_actions else None)

    def run(self, transactions: pd.DataFrame, top_k_features: int = 3) -> List[TransactionOutcome]:
        probs = self.model.predict_proba(transactions)
        outcomes = []
        for i, (idx, row) in enumerate(transactions.iterrows()):
            p = float(probs[i])
            decision = self.decision_layer.decide(p)
            outcome = TransactionOutcome(
                index=idx, amount=float(row["Amount"]), probability=p, action=decision.action,
            )
            if decision.action in self.explain_actions and self.explainer is not None:
                row_df = transactions.iloc[[i]]
                impacts = self.explainer.explain(row_df, top_k=top_k_features)
                summary = self.summarizer.summarize(decision.action, p, outcome.amount, impacts) if self.summarizer else None
                outcome.impacts = impacts
                outcome.summary = summary
            outcomes.append(outcome)
        return outcomes

    def run_one(self, transaction_row: pd.DataFrame, top_k_features: int = 3) -> TransactionOutcome:
        return self.run(transaction_row, top_k_features=top_k_features)[0]
