# Agentic Fraud Detection

A from-scratch implementation of an agentic fraud-detection pipeline:
**predict → decide → act**, instead of a model that just outputs a
probability and stops.

This isn't a from-scratch reimplementation of any specific person's
codebase — it's an original build of the *architecture pattern* described
in the write-up "What Building an Agentic Fraud-Detection System Taught Me
About Agentic AI": a gradient-boosted model scores each transaction, a
cost-based decision layer turns that score into an action, and for
anything flagged, an LLM generates a grounded, plain-English summary for
a human reviewer. It also deliberately reproduces two specific,
counterintuitive findings from that write-up as runnable checks rather
than just prose claims (see "What this reproduces, and why" below).

## Architecture

```
transaction
     │
     ▼
┌─────────────┐   fraud probability   ┌──────────────────┐   action    ┌────────────────────┐
│   PREDICT   │ ───────────────────►  │      DECIDE       │ ──────────► │        ACT          │
│ (src/model) │                       │ (src/decision)     │             │ (src/explain +      │
│ XGBoost /   │                       │ cost-based          │             │  src/summarize)      │
│ sklearn GBC │                       │ two-threshold        │             │ SHAP -> semantic     │
└─────────────┘                       │ review/block layer   │             │ signals -> LLM       │
                                       └──────────────────┘             │ reviewer summary      │
                                                                          └────────────────────┘
```

| Layer | Module | What it does |
|---|---|---|
| Predict | `src/model.py` | Gradient-boosted classifier (XGBoost, or sklearn's `GradientBoostingClassifier` if xgboost isn't installed) scores each transaction. |
| Decide | `src/decision.py` | Turns the score into `approve` / `review` / `block` by minimizing total expected cost (false negatives, false positives, and review cost all weighted differently), not a fixed threshold. |
| Explain | `src/explain.py` | Computes per-transaction SHAP values (or a z-score-based fallback), and converts them into a small vocabulary of semantic signal labels — never raw feature names — before anything reaches the LLM. |
| Act | `src/summarize.py` | Generates a 2-3 sentence reviewer-facing summary from the semantic signals. Supports Groq (Llama 3.3), OpenAI, or Anthropic; falls back to a grounded template generator if no API key is set. |
| Orchestration | `src/pipeline.py` | Wires the above into a single `FraudAgent.run(transactions)` call. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`xgboost` and `shap` are optional — the project runs without them, using
tested fallbacks (see "Design decisions" below). Install them for the
real thing.

For LLM-generated summaries, copy `.env.example` to `.env` and set one
API key (Groq, OpenAI, or Anthropic — checked in that order):

```bash
cp .env.example .env
# edit .env and set e.g. GROQ_API_KEY=...
export $(cat .env | xargs)   # or use python-dotenv / your shell's method
```

Without any key set, `src/summarize.py` uses a template-based fallback
that's still grounded in the actual SHAP signals — no LLM call, no
hallucination risk, just less varied phrasing.

## Run it

```bash
python train.py     # generates synthetic data, trains the model, tunes the
                     # decision layer, runs the cost analyses below, and
                     # saves the model + decision config + test split to models/

python demo.py       # loads the trained model and runs the full
                      # predict -> decide -> act pipeline on sample
                      # transactions, printing decisions + reviewer summaries

python report.py     # generates outputs/report.html -- a self-contained
                      # dashboard with the ROC curve, cost-sweep curve,
                      # business-rule comparison chart, and a table of
                      # sample decisions with reviewer summaries. Open it
                      # in any browser.

python score.py --input data/transactions.csv --output outputs/scored.csv
                      # batch-scores any CSV with V1..V28 + Amount columns
                      # through the full agent and writes a results CSV
                      # (probability, action, top signals, reviewer summary
                      # per row). Add --limit N for a quick test.

python score.py --input data/transactions.csv --output outputs/scored_fast.csv --no-summaries
                   # fast mode: pure probability + decision scoring without
                   # explanation/summarization

python score.py --input data/transactions.csv --output outputs/scored.csv \
              --metrics-output outputs/metrics.json \
              --schema-report-output outputs/schema_report.json \
              --cost-sensitivity-output outputs/cost_sensitivity.json
                   # writes evaluation metrics, schema validation report,
                   # and threshold/cost sensitivity analysis (needs Class)

python benchmark.py --input data/transactions.csv --rows 2000 --output outputs/benchmark.json
                   # compares throughput of full mode vs no-summaries mode
```

If you set an API key in `.env` (see Setup above), all three of
`train.py`, `demo.py`, `report.py`, and `score.py` pick it up
automatically via `python-dotenv` -- no need to `export` it yourself.

`train.py` prints four things worth actually reading, not just skimming:
a single-threshold cost sweep, the tuned two-threshold decision layer,
the "$500 rule" cost-multiplier check, and a "confidently missed fraud"
audit. See below for what each one is checking and why.

Run tests with:

```bash
pip install pytest
pytest tests/ -v
```

### Optional cost matrix scenarios

`score.py` can export threshold/cost sensitivity for multiple business-cost
assumptions in one run via `--cost-matrices`.

Example `cost_matrices.json`:

```json
[
     {
          "name": "default",
          "cost_matrix": {
               "cost_false_negative": 500,
               "cost_false_positive": 25,
               "cost_review": 3,
               "cost_true_positive_extra": 0
          }
     },
     {
          "name": "high_review_cost",
          "cost_matrix": {
               "cost_false_negative": 500,
               "cost_false_positive": 25,
               "cost_review": 10,
               "cost_true_positive_extra": 0
          }
     }
]
```

Then run:

```bash
python score.py --input data/transactions.csv --output outputs/scored.csv \
     --cost-matrices cost_matrices.json \
     --cost-sensitivity-output outputs/cost_sensitivity.json
```

## What this reproduces, and why

The write-up this project is based on made two points that are easy to
state but easy to get wrong in practice. Both are implemented here as
things the code actually checks, not just asserts in a docstring.

**1. Threshold tuning alone can barely move total cost, and that's a
signal, not a bug.** `train.py` runs a cost sweep across single
thresholds and reports how flat or sharp the resulting cost curve is. On
this synthetic dataset (tuned so false negatives cost ~15-30x more than
false positives, similar to the ratio described in the original
write-up), the curve is close to flat — most of the threshold range gives
similar total cost. The reason isn't that the model is bad; it's that a
small number of fraud cases get scored with near-zero confidence
(`_is_confidently_missed` rows in the synthetic dataset, deliberately
built to be statistically indistinguishable from legitimate
transactions). No threshold fixes those — they need better
features/signal, not better decisioning. `train.py`'s final section
("Confidently-wrong audit") surfaces exactly these cases and their dollar
value.

**2. An intuitively "obviously correct" business rule can make total cost
much worse, not better.** `src/decision.py::evaluate_business_rule` and
`tests/test_decision.py::test_amount_business_rule_regression` implement
and regression-test the specific "always flag transactions over $500"
rule from the write-up. The synthetic dataset is deliberately shaped with
a realistic chunk of legitimate big-ticket purchases (`data/generate_data.py`)
so this isn't rigged to fail trivially — it fails for the same underlying
reason the original rule did: a blanket amount cutoff floods reviewers
with false positives on legitimate large purchases, and that flooding
cost outweighs the benefit of the few fraud cases the rule incidentally
catches. On a fresh `python train.py` run you should see roughly a
2-4x cost increase from the rule; the exact number moves with the random
seed and dataset size, which is itself part of the point — it's not a
fixed constant, it's a property of the cost structure and the amount
distribution.

## Design decisions worth knowing about

**Synthetic data, not a downloaded dataset.** The project generates its
own transaction data (`data/generate_data.py`) instead of requiring a
download, so it's fully self-contained and runnable offline from a fresh
clone. The generator is built so a handful of features carry real signal
(with noise / class overlap, not a clean separation), a few "confidently
missed" fraud rows are statistically indistinguishable from legitimate
ones, and legitimate transaction amounts are bimodal (mostly small,
plus a meaningful chunk of legitimate big-ticket purchases) — all
deliberate, and all documented inline in that file.

**Fallback backends everywhere, tested, not just claimed.** `xgboost`
and `shap` are the primary backends because they match the reference
architecture, but both are optional: `src/model.py` falls back to
`sklearn.ensemble.GradientBoostingClassifier`, and `src/explain.py` falls
back to a z-score-weighted local attribution, if the primary packages
aren't installed. Both fallback paths were used to produce the numbers in
this README and pass the same test suite — they're not an untested
afterthought.

**The explainer never hands the LLM raw feature names or raw SHAP
floats.** `src/explain.py` converts SHAP output into
`(semantic_label, direction, relative_strength)` tuples using a fixed,
bounded vocabulary before anything reaches `src/summarize.py`. This
mirrors a specific lesson from the write-up: raw PCA component names
(`V14 = -6.2`) are meaningless to a reviewer, and handing an LLM raw
numbers invites either jargon-dumping or a plausible-sounding but
ungrounded story. The LLM's system prompt also explicitly instructs it
not to invent feature names or claim certainty beyond what it was given.

**One-shot example over "vary your phrasing" instructions.** The system
prompt in `src/summarize.py` includes one concrete worked example rather
than only an instruction to avoid repetitive structure — per the
write-up, telling a model to "vary your phrasing" tends not to work
reliably, while giving it one real pattern to riff on does better.

**Provider order: Groq, then OpenAI, then Anthropic.** `src/summarize.py`
checks environment variables in that order and uses whichever key is
present. This follows the write-up's account of one free-tier provider
becoming unstable mid-project and Groq's Llama 3.3 tier being more
reliable — that's a practical/cost note about free-tier availability, not
a claim about model quality, and the module works the same way with any
of the three.

## What's intentionally left as an honest gap

The write-up this is based on is explicit that it never rigorously
verified the LLM summaries are never plausible-but-wrong — SHAP grounding
reduces that risk but doesn't eliminate it, and doing so properly needs a
human checking summaries against real transaction details at scale. This
project doesn't solve that either. If you wire up a real LLM provider,
treat `tests/test_explain_and_summarize.py` as a starting point, not
proof the summaries are always faithful — it checks groundedness of the
template fallback and the absence of raw feature-name leakage, not
whether an LLM-generated summary is always accurate.

## Project layout

```
fraud-agent/
├── data/
│   └── generate_data.py     # synthetic transaction dataset generator
├── src/
│   ├── model.py              # PREDICT: XGBoost / sklearn fallback
│   ├── decision.py           # DECIDE: cost-based two-threshold layer
│   ├── explain.py            # SHAP -> semantic signal labels
│   ├── summarize.py          # ACT: LLM (or template fallback) summaries
│   └── pipeline.py           # orchestrates predict -> decide -> act
├── tests/
│   ├── test_decision.py
│   └── test_explain_and_summarize.py
├── train.py                  # generate data, train, tune, save artifacts
├── demo.py                   # run the full agent on sample transactions
├── report.py                 # HTML dashboard: charts + sample decisions
├── score.py                  # batch-score an arbitrary transactions CSV
├── requirements.txt
├── .env.example
└── README.md
```
