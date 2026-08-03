import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generate_data import generate
from src.model import FraudModel
from src.decision import CostBasedDecisionLayer, CostMatrix, Action, evaluate_business_rule


def _small_trained_model():
    df = generate(n_legit=2000, n_fraud=60, n_confidently_missed=2, seed=7)
    model = FraudModel()
    result = model.fit(df, test_size=0.3, seed=7)
    return model, result


def test_decide_thresholds():
    layer = CostBasedDecisionLayer(threshold_review=0.3, threshold_block=0.8)
    assert layer.decide(0.1).action == Action.APPROVE
    assert layer.decide(0.5).action == Action.REVIEW
    assert layer.decide(0.95).action == Action.BLOCK
    # boundary conditions
    assert layer.decide(0.3).action == Action.REVIEW
    assert layer.decide(0.8).action == Action.BLOCK


def test_total_cost_is_nonnegative_and_monotonic_in_fraud_cost():
    layer_cheap = CostBasedDecisionLayer(0.5, 0.5, CostMatrix(cost_false_negative=10))
    layer_expensive = CostBasedDecisionLayer(0.5, 0.5, CostMatrix(cost_false_negative=1000))
    y = np.array([1, 0, 1, 0])
    p = np.array([0.1, 0.1, 0.1, 0.1])  # all approved -> both false negatives count
    cost_cheap = layer_cheap.total_cost(y, p)
    cost_expensive = layer_expensive.total_cost(y, p)
    assert cost_cheap >= 0 and cost_expensive >= 0
    assert cost_expensive > cost_cheap


def test_total_cost_matches_manual_action_accounting():
    layer = CostBasedDecisionLayer(
        threshold_review=0.4,
        threshold_block=0.8,
        cost_matrix=CostMatrix(
            cost_false_negative=500,
            cost_false_positive=25,
            cost_review=3,
            cost_true_positive_extra=2,
        ),
    )
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([0.2, 0.2, 0.6, 0.6, 0.95, 0.95])

    # approve fraud: 1 * 500
    # review total: 2 * 3 and review fraud extra: 1 * 2
    # block legit: 1 * 25 and block fraud extra: 1 * 2
    manual = 500 + 6 + 2 + 25 + 2
    assert layer.total_cost(y, p) == manual


def test_fit_thresholds_does_not_increase_cost_vs_default():
    model, _ = _small_trained_model()
    y_test, proba_test = model._y_test.values, model._proba_test
    layer = CostBasedDecisionLayer(0.5, 0.5)
    cost_before = layer.total_cost(y_test, proba_test)
    layer.fit_thresholds(y_test, proba_test)
    cost_after = layer.total_cost(y_test, proba_test)
    assert cost_after <= cost_before


def test_amount_business_rule_regression():
    """This is the core lesson the project is built around: an intuitive
    'flag every transaction over $X' rule should NOT be assumed safe.
    On this synthetic dataset (deliberately shaped with a chunk of
    legitimate big-ticket purchases) the rule should make total cost
    worse than the tuned cost-based baseline, not better.

    Uses a larger sample than the other tests here: the flooding effect
    this checks for needs enough big-ticket legitimate transactions in
    the test split to show up reliably rather than being dominated by
    the luck of a single high-value miss."""
    df = generate(n_legit=10_000, n_fraud=200, n_confidently_missed=3, seed=11)
    model = FraudModel()
    model.fit(df, test_size=0.3, seed=11)
    y_test, proba_test = model._y_test.values, model._proba_test
    amounts_test = model._X_test["Amount"].values

    layer = CostBasedDecisionLayer(0.5, 0.5)
    layer.fit_thresholds(y_test, proba_test)

    result = evaluate_business_rule(y_test, proba_test, amounts_test, layer, amount_threshold=500.0)
    assert result["rule_cost"] >= result["base_cost"]
    assert result["multiplier"] >= 1.0
