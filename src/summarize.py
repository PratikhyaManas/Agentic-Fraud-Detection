"""
summarize.py -- the ACT layer.

For every transaction that gets flagged for review or blocked, this module
produces a short, plain-English explanation a human reviewer can act on
immediately, instead of a bare probability. This is the piece that makes
the system "agentic" rather than "a model with a threshold on top": it
produces an artifact, not just a score.

Two lessons from building the reference version of this shaped the design:

1. The failure mode to worry about is repetitive template-following, not
   hallucination. Every summary technically-accurately following the exact
   same "one dominant anomaly, two secondary factors, Nx normal" structure
   reads as templated rather than case-specific, even when every sentence
   is true. Telling the model to "vary phrasing" in the instructions did
   not fix this reliably -- what worked was giving the model one concrete
   worked example (one-shot) so it has a real pattern to riff on rather
   than an abstract instruction to vary something.

2. Grounding matters more than fluency. The model is given only the
   *semantic label + direction + relative strength* of the top SHAP
   features (see explain.py) -- never raw feature names or raw SHAP
   floats -- with an explicit instruction not to invent feature names or
   claim certainty the evidence doesn't support. This bounds the model to
   describing what the evidence actually shows.

Provider notes: this module supports Groq (Llama 3.3), OpenAI, and
Anthropic, selected by whichever API key is present in the environment,
checked in that order. Groq's free tier is used first because -- per the
same lesson learned building the reference system -- some providers'
free tiers get deprecated or rate-limited with little warning, and Groq's
Llama 3.3 was found to be both more stable and, as a side effect, less
repetitive than the alternative tried first. If no API key is present at
all, a template-based fallback generates a grounded (if less varied)
summary so the pipeline still produces useful output offline.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional

from .explain import FeatureImpact
from .decision import Action

ONE_SHOT_EXAMPLE = """Example transaction and the kind of summary we want:

Signals: spending velocity (elevated, 3.2x typical influence), \
geographic consistency (suppressed, 2.1x typical influence), \
transaction amount (elevated, 1.4x typical influence)
Action: review

Good summary: "This purchase came in well outside the account's normal \
rhythm -- a burst of spending activity paired with a location pattern \
that doesn't match the account's usual footprint. The dollar amount adds \
some weight but isn't the main driver here; the timing and location \
mismatch are what pushed this into review."
"""

SYSTEM_PROMPT = f"""You are writing a short explanation for a fraud-review \
analyst, based on a transaction's model score and its top contributing \
signals. You will be given only semantic signal labels (e.g. "spending \
velocity"), a direction ("elevated" or "suppressed"), and a relative \
strength -- never raw feature names or raw model internals. Do not invent \
feature names, PCA component names, or any statistic you were not given. \
Do not claim certainty the evidence doesn't support -- describe what the \
signals suggest, not a verdict. Vary your sentence structure and opening \
between summaries; do not default to a fixed template of "one dominant \
factor, two secondary factors." Two to three sentences. Plain English, no \
jargon, no bullet points.

{ONE_SHOT_EXAMPLE}
"""


@dataclass
class Summary:
    text: str
    provider: str


def _build_user_prompt(action: Action, probability: float, amount: float,
                        impacts: List[FeatureImpact]) -> str:
    signal_lines = "\n".join(
        f"- {imp.semantic_label} ({imp.direction}, {imp.strength_label})"
        for imp in impacts
    )
    return (
        f"Transaction amount: ${amount:,.2f}\n"
        f"Model fraud probability: {probability:.1%}\n"
        f"Decision: {action.value}\n"
        f"Top contributing signals:\n{signal_lines}\n\n"
        f"Write the reviewer-facing summary now."
    )


class Summarizer:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider, self.model = self._resolve_provider(provider, model)

    @staticmethod
    def _resolve_provider(provider, model):
        if provider:
            return provider, model
        if os.environ.get("GROQ_API_KEY"):
            return "groq", model or "llama-3.3-70b-versatile"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai", model or "gpt-4o-mini"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic", model or "claude-sonnet-4-6"
        return "template_fallback", None

    # ------------------------------------------------------------- public
    def summarize(self, action: Action, probability: float, amount: float,
                  impacts: List[FeatureImpact]) -> Summary:
        if self.provider == "groq":
            return self._call_groq(action, probability, amount, impacts)
        if self.provider == "openai":
            return self._call_openai(action, probability, amount, impacts)
        if self.provider == "anthropic":
            return self._call_anthropic(action, probability, amount, impacts)
        return self._template_fallback(action, probability, amount, impacts)

    # -------------------------------------------------------- providers
    def _call_groq(self, action, probability, amount, impacts) -> Summary:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(action, probability, amount, impacts)},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        return Summary(resp.choices[0].message.content.strip(), "groq:" + self.model)

    def _call_openai(self, action, probability, amount, impacts) -> Summary:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(action, probability, amount, impacts)},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        return Summary(resp.choices[0].message.content.strip(), "openai:" + self.model)

    def _call_anthropic(self, action, probability, amount, impacts) -> Summary:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=self.model,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(action, probability, amount, impacts)}],
        )
        return Summary(resp.content[0].text.strip(), "anthropic:" + self.model)

    # ----------------------------------------------------------- fallback
    def _template_fallback(self, action, probability, amount, impacts) -> Summary:
        """Grounded, varied-ish summary with no LLM call. Used when no API
        key is configured. Several sentence templates are rotated (keyed
        off the transaction so it's deterministic, not random per run) to
        avoid the exact "always the same shape" problem the fallback would
        otherwise reproduce.
        """
        top = impacts[0] if impacts else None
        rest = impacts[1:] if len(impacts) > 1 else []

        seed = int(abs(probability) * 10_000) + int(amount)
        rng = random.Random(seed)

        openers = [
            "This transaction stood out primarily because of {top}, which came in {dir_top}.",
            "The main driver flagged here is {top}, showing a {dir_top} pattern relative to this account's norm.",
            "What pushed this transaction to {action} was {top} coming in {dir_top}.",
        ]
        opener = rng.choice(openers).format(
            top=top.semantic_label if top else "the overall signal mix",
            dir_top=top.direction if top else "unusual",
            action=action.value,
        )

        if rest:
            extra_desc = " and ".join(f"{imp.semantic_label} ({imp.direction})" for imp in rest)
            body = f" Supporting signals include {extra_desc}, though these carried less weight than the primary factor."
        else:
            body = ""

        closer_map = {
            Action.BLOCK: f" The model's fraud probability was {probability:.1%} on a ${amount:,.2f} transaction, high enough to block automatically pending appeal.",
            Action.REVIEW: f" At a {probability:.1%} fraud probability on a ${amount:,.2f} transaction, this falls in the range worth a quick manual look rather than an automatic decision.",
            Action.APPROVE: f" At {probability:.1%}, this is within the account's normal range and was approved automatically.",
        }
        closer = closer_map[action]

        text = opener + body + closer
        return Summary(text, "template_fallback")
