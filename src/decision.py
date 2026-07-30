"""
Decide layer: cost-sensitive action selection.

Maps a fraud probability into one of three actions:
  - APPROVE
  - FLAG   (send to human review)
  - BLOCK

Uses explicit false-positive / false-negative costs instead of a fixed threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class Action(str, Enum):
    APPROVE = "APPROVE"
    FLAG = "FLAG"
    BLOCK = "BLOCK"


@dataclass
class CostConfig:
    """
    Business costs (in arbitrary currency units).

    Typical fraud domain:
      - False negative (missed fraud) is 20–50× more expensive than a false positive.
      - Flagging incurs a small human-review cost.
    """

    cost_fp: float = 10.0          # cost of reviewing / inconveniencing a legit customer
    cost_fn: float = 350.0         # cost of a missed fraud (chargeback + ops)
    cost_review: float = 5.0       # cost of sending a transaction to a human
    cost_block_legit: float = 25.0 # higher friction cost when we hard-block a legit txn

    # Probability bands that map to actions (can be tuned)
    block_threshold: float = 0.85
    flag_threshold: float = 0.25


@dataclass
class DecisionResult:
    action: Action
    fraud_probability: float
    expected_cost_approve: float
    expected_cost_flag: float
    expected_cost_block: float
    rationale: str


class CostBasedDecisionMaker:
    """
    Chooses the action that minimises expected cost under the assumed
    cost matrix, with optional hard probability floors/ceilings.
    """

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()

    def decide(self, p_fraud: float, amount: float = 0.0) -> DecisionResult:
        """
        Compute expected cost of each action and pick the cheapest.

        Expected costs (simplified):
          APPROVE : p * cost_fn * (amount scaling) + (1-p) * 0
          FLAG    : cost_review + p * (small residual FN) + (1-p) * cost_fp * 0.3
          BLOCK   : (1-p) * cost_block_legit + p * 0
        """
        cfg = self.config
        # Scale FN cost mildly with transaction amount (log to avoid explosion)
        amount_factor = 1.0 + 0.15 * max(0.0, (amount / 100.0) ** 0.5)

        exp_approve = p_fraud * cfg.cost_fn * amount_factor
        exp_flag = (
            cfg.cost_review
            + p_fraud * cfg.cost_fn * 0.15 * amount_factor   # residual risk after review
            + (1 - p_fraud) * cfg.cost_fp * 0.4
        )
        exp_block = (1 - p_fraud) * cfg.cost_block_legit

        # Soft thresholds still useful as guardrails
        if p_fraud >= cfg.block_threshold:
            action = Action.BLOCK
            rationale = (
                f"Probability {p_fraud:.3f} exceeds hard block threshold "
                f"({cfg.block_threshold}). Expected cost of blocking is lowest."
            )
        elif p_fraud >= cfg.flag_threshold:
            # Among the three, pick min expected cost, but bias toward FLAG
            # when probability is in the grey zone
            costs = {
                Action.APPROVE: exp_approve,
                Action.FLAG: exp_flag,
                Action.BLOCK: exp_block,
            }
            action = min(costs, key=costs.get)
            rationale = (
                f"Grey-zone probability {p_fraud:.3f}. "
                f"Min expected cost action selected "
                f"(approve={exp_approve:.1f}, flag={exp_flag:.1f}, block={exp_block:.1f})."
            )
        else:
            action = Action.APPROVE
            rationale = (
                f"Probability {p_fraud:.3f} below flag threshold "
                f"({cfg.flag_threshold}). Approving."
            )

        return DecisionResult(
            action=action,
            fraud_probability=p_fraud,
            expected_cost_approve=exp_approve,
            expected_cost_flag=exp_flag,
            expected_cost_block=exp_block,
            rationale=rationale,
        )

    def decide_batch(
        self, probabilities: list[float], amounts: list[float]
    ) -> list[DecisionResult]:
        return [
            self.decide(p, a) for p, a in zip(probabilities, amounts)
        ]