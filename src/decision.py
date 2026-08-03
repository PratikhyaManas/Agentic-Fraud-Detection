"""
decision.py -- the DECIDE layer.

Turns a fraud probability into an actual action: approve, review, or block.
The naive version of this is a fixed probability threshold. The version
here is cost-based: it weighs the real cost of each outcome (approving a
fraud, blocking a legitimate purchase, sending something to manual review)
and picks the threshold that minimizes total expected cost -- because in
most real fraud programs, false negatives (missed fraud) and false
positives (annoyed legitimate customers / wasted reviewer time) are not
equally expensive, and a single "accuracy-optimal" threshold usually isn't
the cost-optimal one.

This module also includes `evaluate_business_rule`, used to demonstrate
a specific, important lesson: an intuitively "obviously correct" rule
("always flag transactions over $500") can look reasonable and still make
total cost dramatically worse, by flooding reviewers with false positives
on legitimate large purchases. The rule is worth testing explicitly and
keeping in the codebase as a regression check, not just a one-off
observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
import pandas as pd


class Action(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"


@dataclass
class CostMatrix:
    """Cost of each outcome, in whatever unit you want (e.g. USD).

    cost_false_negative: fraud approved and missed entirely.
    cost_false_positive: legitimate transaction blocked outright.
    cost_review: cost of sending a transaction to manual review
                 (reviewer time), charged regardless of the true label.
    cost_true_positive_extra: any *additional* cost of a correctly caught
                 fraud beyond the review cost (e.g. chargeback handling
                 that still occurs even when caught late). Defaults to 0.
    """
    cost_false_negative: float = 500.0
    cost_false_positive: float = 25.0
    cost_review: float = 3.0
    cost_true_positive_extra: float = 0.0


@dataclass
class DecisionResult:
    action: Action
    probability: float
    threshold_review: float
    threshold_block: float


def _total_cost_from_action_masks(
    y_true: np.ndarray,
    approve_mask: np.ndarray,
    review_mask: np.ndarray,
    block_mask: np.ndarray,
    cost_matrix: CostMatrix,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    cm = cost_matrix

    approve_fraud = np.sum(approve_mask & (y_true == 1))
    review_count = np.sum(review_mask)
    review_fraud = np.sum(review_mask & (y_true == 1))
    block_legit = np.sum(block_mask & (y_true == 0))
    block_fraud = np.sum(block_mask & (y_true == 1))

    total = 0.0
    total += cm.cost_false_negative * approve_fraud
    total += cm.cost_review * review_count
    total += cm.cost_true_positive_extra * review_fraud
    total += cm.cost_false_positive * block_legit
    total += cm.cost_true_positive_extra * block_fraud
    return float(total)


class CostBasedDecisionLayer:
    """Two-threshold decisioning: below `threshold_review` -> approve,
    between the two thresholds -> review, above `threshold_block` -> block.

    Two thresholds (rather than one) let the system distinguish "worth a
    human look" from "confident enough to act automatically" -- collapsing
    to a single threshold is possible by setting the two equal.
    """

    def __init__(self, threshold_review: float = 0.5, threshold_block: float = 0.9,
                 cost_matrix: Optional[CostMatrix] = None):
        self.threshold_review = threshold_review
        self.threshold_block = threshold_block
        self.cost_matrix = cost_matrix or CostMatrix()

    def decide(self, probability: float) -> DecisionResult:
        if probability >= self.threshold_block:
            action = Action.BLOCK
        elif probability >= self.threshold_review:
            action = Action.REVIEW
        else:
            action = Action.APPROVE
        return DecisionResult(action, probability, self.threshold_review, self.threshold_block)

    def decide_batch(self, probabilities: np.ndarray) -> list[Action]:
        return [self.decide(p).action for p in probabilities]

    # ---------------------------------------------------------- cost math
    def total_cost(self, y_true: np.ndarray, probabilities: np.ndarray) -> float:
        probabilities = np.asarray(probabilities)
        approve_mask = probabilities < self.threshold_review
        review_mask = (probabilities >= self.threshold_review) & (probabilities < self.threshold_block)
        block_mask = probabilities >= self.threshold_block
        return _total_cost_from_action_masks(
            y_true,
            approve_mask,
            review_mask,
            block_mask,
            self.cost_matrix,
        )

    # --------------------------------------------------------- calibration
    def sweep_single_threshold(
        self, y_true: np.ndarray, probabilities: np.ndarray,
        grid: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Collapse review==block threshold and sweep it, for the classic
        "what's the cost-minimizing cutoff" analysis. Returns a DataFrame
        of threshold -> total_cost so the curve (often surprisingly flat)
        can be inspected directly instead of only reporting the argmin.
        """
        grid = grid if grid is not None else np.linspace(0.01, 0.99, 99)
        rows = []
        for t in grid:
            layer = CostBasedDecisionLayer(t, t, self.cost_matrix)
            rows.append({"threshold": t, "total_cost": layer.total_cost(y_true, probabilities)})
        return pd.DataFrame(rows)

    def fit_thresholds(
        self, y_true: np.ndarray, probabilities: np.ndarray,
        review_grid: Optional[np.ndarray] = None,
        block_grid: Optional[np.ndarray] = None,
    ) -> "CostBasedDecisionLayer":
        """Grid search over (threshold_review, threshold_block) pairs for
        the pair that minimizes total cost. Small grids by default since
        this is O(grid^2 * n)."""
        review_grid = review_grid if review_grid is not None else np.linspace(0.05, 0.6, 12)
        block_grid = block_grid if block_grid is not None else np.linspace(0.6, 0.99, 12)

        best = (self.threshold_review, self.threshold_block, float("inf"))
        for tr in review_grid:
            for tb in block_grid:
                if tb < tr:
                    continue
                layer = CostBasedDecisionLayer(tr, tb, self.cost_matrix)
                cost = layer.total_cost(y_true, probabilities)
                if cost < best[2]:
                    best = (tr, tb, cost)
        self.threshold_review, self.threshold_block, _ = best
        return self


def evaluate_business_rule(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    base_layer: CostBasedDecisionLayer,
    amount_threshold: float = 500.0,
) -> dict:
    """Reproduces the specific cautionary result this project is based on:
    an "always flag transactions over $X" rule stacked on top of an
    already-reasonable cost-based layer. Returns both costs so the
    regression is explicit rather than asserted.
    """
    base_cost = base_layer.total_cost(y_true, probabilities)

    probabilities = np.asarray(probabilities)
    amounts = np.asarray(amounts)
    forced_review = amounts > amount_threshold

    approve_mask = (probabilities < base_layer.threshold_review) & ~forced_review
    block_mask = (probabilities >= base_layer.threshold_block) & ~forced_review
    review_mask = ~(approve_mask | block_mask)

    rule_cost = _total_cost_from_action_masks(
        y_true,
        approve_mask,
        review_mask,
        block_mask,
        base_layer.cost_matrix,
    )

    return {
        "base_cost": base_cost,
        "rule_cost": rule_cost,
        "multiplier": rule_cost / base_cost if base_cost > 0 else float("inf"),
        "amount_threshold": amount_threshold,
    }
