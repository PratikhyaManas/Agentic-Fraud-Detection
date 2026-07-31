import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generate_data import generate
from src.model import FraudModel, FEATURE_COLUMNS
from src.explain import Explainer
from src.summarize import Summarizer
from src.decision import Action


def _fitted():
    df = generate(n_legit=1500, n_fraud=40, n_confidently_missed=1, seed=3)
    model = FraudModel()
    model.fit(df, test_size=0.3, seed=3)
    return model, df


def test_explainer_returns_topk_unique_labels():
    model, df = _fitted()
    background = df.sample(200, random_state=1)
    explainer = Explainer(model, background)
    row = df.iloc[[0]]
    impacts = explainer.explain(row, top_k=3)
    assert len(impacts) == 3
    labels = [imp.semantic_label for imp in impacts]
    assert len(labels) == len(set(labels))  # no duplicate signal labels
    for imp in impacts:
        assert imp.direction in ("elevated", "suppressed")


def test_explainer_never_leaks_raw_feature_names():
    model, df = _fitted()
    background = df.sample(200, random_state=1)
    explainer = Explainer(model, background)
    row = df.iloc[[0]]
    impacts = explainer.explain(row, top_k=5)
    for imp in impacts:
        assert imp.semantic_label not in FEATURE_COLUMNS
        assert not imp.semantic_label.startswith("V")


def test_summarizer_template_fallback_is_grounded():
    model, df = _fitted()
    background = df.sample(200, random_state=1)
    explainer = Explainer(model, background)
    row = df.iloc[[0]]
    impacts = explainer.explain(row, top_k=3)

    summarizer = Summarizer(provider="template_fallback")
    summary = summarizer.summarize(Action.REVIEW, 0.62, 123.45, impacts)
    assert summary.text
    assert "123.45" in summary.text or "$123" in summary.text
    # should reference at least the top semantic label, not a raw feature name
    assert impacts[0].semantic_label in summary.text
