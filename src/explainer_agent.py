"""
Act layer: generate a human-readable reviewer summary.

In production this would call an LLM (Groq / Gemini / OpenAI).
Because the sandbox has no outbound LLM API access, we implement a
high-quality *template + variation* generator that stays strictly
grounded in the SHAP impacts — the same constraint the original article
emphasised.

The interface is deliberately identical to what an LLM wrapper would
expose, so swapping in a real model later is a one-line change.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from .decision import Action, DecisionResult


# Human-friendly feature labels (never expose raw PCA / internal names)
FEATURE_LABELS = {
    "amount": "transaction amount",
    "hour_of_day": "time of day",
    "day_of_week": "day of week",
    "distance_from_home_km": "distance from the cardholder's home",
    "distance_from_last_txn_km": "distance from the previous transaction",
    "ratio_to_median_purchase_price": "ratio to the cardholder's typical spend",
    "repeat_retailer": "whether this is a merchant the customer has used before",
    "used_chip": "whether the physical chip was used",
    "used_pin_number": "whether a PIN was entered",
    "online_order": "whether the order was placed online",
}


class ReviewerSummaryAgent:
    """
    Produces a short, grounded, natural-language explanation for a human
    fraud analyst.  Stays faithful to SHAP directions and never invents
    features.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(
        self,
        transaction_id: str,
        amount: float,
        decision: DecisionResult,
        shap_impacts: List[Dict[str, Any]],
    ) -> str:
        """
        Build a reviewer-facing summary.

        Parameters
        ----------
        transaction_id : str
        amount : float
        decision : DecisionResult from the Decide layer
        shap_impacts : list of dicts from FraudModel.explain_transaction
                       (already sorted by absolute impact)
        """
        if decision.action == Action.APPROVE:
            return (
                f"Transaction {transaction_id} (${amount:,.2f}) was auto-approved. "
                f"Fraud probability {decision.fraud_probability:.1%} is below the "
                f"review threshold. No analyst action required."
            )

        # Only FLAG / BLOCK need rich explanations
        risk_drivers = [i for i in shap_impacts if i["direction"] == "increases_fraud_risk"]
        protective = [i for i in shap_impacts if i["direction"] == "decreases_fraud_risk"]

        opener = self._pick_opener(decision.action, amount, decision.fraud_probability)
        body = self._describe_drivers(risk_drivers[:3])
        mitigator = self._describe_mitigators(protective[:1])
        closer = self._pick_closer(decision.action)

        parts = [opener, body]
        if mitigator:
            parts.append(mitigator)
        parts.append(closer)
        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Private helpers – variation comes from random choice of templates
    # ------------------------------------------------------------------

    def _pick_opener(self, action: Action, amount: float, p: float) -> str:
        verb = "blocked" if action == Action.BLOCK else "flagged for review"
        templates = [
            f"Transaction of ${amount:,.2f} has been {verb} (model score {p:.1%}).",
            f"A ${amount:,.2f} payment was {verb} because the fraud probability reached {p:.1%}.",
            f"Review recommended: ${amount:,.2f} transaction scored {p:.1%} risk and was {verb}.",
        ]
        return self.rng.choice(templates)

    def _describe_drivers(self, drivers: List[Dict[str, Any]]) -> str:
        if not drivers:
            return "No dominant risk signals were isolated by the explainer."

        phrases = []
        for i, d in enumerate(drivers):
            label = FEATURE_LABELS.get(d["feature"], d["feature"].replace("_", " "))
            val = d["value"]
            # Light natural-language formatting of the value
            if d["feature"] == "amount":
                val_str = f"${val:,.0f}"
            elif "distance" in d["feature"]:
                val_str = f"{val:.0f} km"
            elif d["feature"] in ("repeat_retailer", "used_chip", "used_pin_number", "online_order"):
                val_str = "yes" if val >= 0.5 else "no"
            elif d["feature"] == "hour_of_day":
                val_str = f"{int(val):02d}:00"
            else:
                val_str = f"{val:.2f}"

            strength = "strongly" if d["abs_impact"] > 0.15 else "moderately"
            if i == 0:
                phrases.append(
                    f"The primary driver is {label} ({val_str}), which {strength} elevates risk."
                )
            else:
                phrases.append(
                    f"Secondary signal: {label} = {val_str}."
                )
        return " ".join(phrases)

    def _describe_mitigators(self, protective: List[Dict[str, Any]]) -> str:
        if not protective:
            return ""
        d = protective[0]
        label = FEATURE_LABELS.get(d["feature"], d["feature"].replace("_", " "))
        templates = [
            f"One mitigating factor is {label}, which slightly lowers the score.",
            f"Note that {label} works in the customer's favour and partially offsets the risk.",
        ]
        return self.rng.choice(templates)

    def _pick_closer(self, action: Action) -> str:
        if action == Action.BLOCK:
            options = [
                "Hard block applied; customer will need to contact support to proceed.",
                "Transaction declined at authorisation. Manual override available if needed.",
            ]
        else:
            options = [
                "Please review the full transaction context and decide within the SLA window.",
                "Analyst review requested — approve, escalate, or confirm block.",
            ]
        return self.rng.choice(options)


# ---------------------------------------------------------------------------
# Drop-in real-LLM adapter (commented; enable when API keys + network exist)
# ---------------------------------------------------------------------------
#
# import os
# from openai import OpenAI   # or groq, google.generativeai, etc.
#
# class LLMReviewerSummaryAgent(ReviewerSummaryAgent):
#     def __init__(self, model: str = "llama-3.3-70b-versatile"):
#         self.client = OpenAI(
#             base_url="https://api.groq.com/openai/v1",
#             api_key=os.environ["GROQ_API_KEY"],
#         )
#         self.model = model
#
#     def generate(self, transaction_id, amount, decision, shap_impacts):
#         # Build a tightly constrained prompt that only receives
#         # direction-of-impact, never raw feature names that could leak.
#         ...
#         response = self.client.chat.completions.create(...)
#         return response.choices[0].message.content