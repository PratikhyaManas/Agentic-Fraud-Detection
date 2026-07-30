"""
Orchestrator: the full Predict → Decide → Act loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_generator import FEATURE_NAMES
from .decision import Action, CostBasedDecisionMaker, CostConfig, DecisionResult
from .explainer_agent import ReviewerSummaryAgent
from .model import FraudModel


@dataclass
class AgentOutput:
    transaction_id: str
    amount: float
    fraud_probability: float
    action: str
    decision_rationale: str
    expected_costs: Dict[str, float]
    reviewer_summary: str
    top_shap_impacts: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FraudDetectionAgent:
    """
    End-to-end agentic fraud system.

    Usage
    -----
    agent = FraudDetectionAgent.from_pretrained("models")
    result = agent.process(transaction_row)   # single pandas Series / 1-row DF
    """

    def __init__(
        self,
        model: FraudModel,
        decision_maker: Optional[CostBasedDecisionMaker] = None,
        summary_agent: Optional[ReviewerSummaryAgent] = None,
    ):
        self.model = model
        self.decision_maker = decision_maker or CostBasedDecisionMaker()
        self.summary_agent = summary_agent or ReviewerSummaryAgent()

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "FraudDetectionAgent":
        model = FraudModel().load(model_dir)
        return cls(model=model)

    def process(self, txn: pd.Series | pd.DataFrame) -> AgentOutput:
        """Run the full pipeline on one transaction."""
        if isinstance(txn, pd.Series):
            row = txn.to_frame().T
        else:
            row = txn.copy()
            assert len(row) == 1

        # Ensure numeric dtypes (Series → DataFrame can promote columns to object)
        for col in FEATURE_NAMES:
            if col in row.columns:
                row[col] = pd.to_numeric(row[col], errors="coerce")

        txn_id = str(row["transaction_id"].iloc[0]) if "transaction_id" in row.columns else "UNKNOWN"
        amount = float(row["amount"].iloc[0])

        # 1. Predict
        p_fraud = float(self.model.predict_proba(row)[0])
        shap_impacts = self.model.explain_transaction(row, top_k=5)

        # 2. Decide
        decision: DecisionResult = self.decision_maker.decide(p_fraud, amount)

        # 3. Act
        summary = self.summary_agent.generate(
            transaction_id=txn_id,
            amount=amount,
            decision=decision,
            shap_impacts=shap_impacts,
        )

        return AgentOutput(
            transaction_id=txn_id,
            amount=amount,
            fraud_probability=p_fraud,
            action=decision.action.value,
            decision_rationale=decision.rationale,
            expected_costs={
                "approve": decision.expected_cost_approve,
                "flag": decision.expected_cost_flag,
                "block": decision.expected_cost_block,
            },
            reviewer_summary=summary,
            top_shap_impacts=shap_impacts,
        )

    def process_batch(self, df: pd.DataFrame) -> List[AgentOutput]:
        return [self.process(df.iloc[[i]]) for i in range(len(df))]