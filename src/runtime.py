from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .decision import CostBasedDecisionLayer, CostMatrix
from .explain import Explainer
from .model import FEATURE_COLUMNS, FraudModel
from .pipeline import FraudAgent
from .summarize import Summarizer


def maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def load_model_and_decision_layer(
    model_path: Path,
    config_path: Path,
) -> tuple[FraudModel, CostBasedDecisionLayer, dict]:
    model = FraudModel.load(model_path)
    config = json.loads(config_path.read_text())
    layer = CostBasedDecisionLayer(
        config["threshold_review"],
        config["threshold_block"],
        CostMatrix(**config["cost_matrix"]),
    )
    return model, layer, config


def missing_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c not in df.columns]


def build_agent(
    model: FraudModel,
    decision_layer: CostBasedDecisionLayer,
    background: Optional[pd.DataFrame],
    summarizer: Optional[Summarizer] = None,
    explain_actions=(),
    background_sample_size: int = 500,
) -> FraudAgent:
    explain_actions = tuple(explain_actions)
    explainer = None
    if explain_actions:
        if background is None or len(background) == 0:
            raise ValueError("A non-empty background DataFrame is required when explanations are enabled.")
        sample_n = min(background_sample_size, len(background))
        explainer = Explainer(model, background.sample(sample_n, random_state=1))
    return FraudAgent(
        model=model,
        decision_layer=decision_layer,
        explainer=explainer,
        summarizer=summarizer,
        explain_actions=explain_actions,
    )