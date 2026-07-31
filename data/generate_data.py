"""
generate_data.py
-----------------
Generates a synthetic transaction dataset that mimics the structure of the
classic anonymized credit-card-fraud dataset: a `Time` column, 28 PCA-style
components (V1..V28), a transaction `Amount`, and a binary `Class` label
(1 = fraud, 0 = legitimate).

Why synthetic instead of downloading a real dataset: this keeps the project
self-contained and runnable offline / from a fresh clone with no external
downloads, licensing, or size concerns. The generator is built so that a
handful of the V-components are genuinely predictive of fraud (with added
noise), some are mildly predictive, and the rest are pure noise -- which
is a reasonable stand-in for what PCA components of real transaction
features tend to look like.

It also deliberately injects a small number of "confidently missed" fraud
cases: fraud transactions engineered to look statistically identical to
legitimate ones. These exist to reproduce a specific, important failure
mode discussed in the write-up this project is based on: cases the model
scores with near-zero fraud probability that no threshold or business rule
can fix, because the model was never close to catching them in the first
place. Keeping a few of these in the dataset is what makes the cost
analysis in `train.py` meaningful instead of trivially "solvable" by
threshold tuning.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

N_COMPONENTS = 28
# Indices (0-based, i.e. V1..V28) that carry real fraud signal, with
# decreasing strength. Everything else is noise.
STRONG_SIGNAL_IDX = [3, 10, 13]     # V4, V11, V14
MEDIUM_SIGNAL_IDX = [1, 6, 16, 20]  # V2, V7, V17, V21


def _make_legit_rows(n: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(loc=0.0, scale=1.0, size=(n, N_COMPONENTS))
    return X


def _make_fraud_rows(n: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(loc=0.0, scale=1.0, size=(n, N_COMPONENTS))
    # Push the "signal" components in a consistent direction for fraud,
    # with noise so the classes overlap (real fraud detection is never
    # a clean separation).
    for idx in STRONG_SIGNAL_IDX:
        X[:, idx] += rng.normal(loc=-4.5, scale=1.5, size=n)
    for idx in MEDIUM_SIGNAL_IDX:
        X[:, idx] += rng.normal(loc=2.5, scale=1.8, size=n)
    return X


def _make_confidently_missed_fraud(n: int, rng: np.random.Generator) -> np.ndarray:
    """Fraud rows statistically indistinguishable from legit rows.

    These represent the "confidently wrong" failure mode: the model will
    score these with near-zero fraud probability because nothing in the
    feature distribution separates them from legitimate transactions.
    """
    return rng.normal(loc=0.0, scale=1.0, size=(n, N_COMPONENTS))


def generate(
    n_legit: int = 10_000,
    n_fraud: int = 200,
    n_confidently_missed: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    legit_X = _make_legit_rows(n_legit, rng)
    fraud_X = _make_fraud_rows(n_fraud - n_confidently_missed, rng)
    missed_X = _make_confidently_missed_fraud(n_confidently_missed, rng)

    # Amounts: a mixture of everyday small purchases and legitimate
    # big-ticket purchases (electronics, travel, rent, etc). Real spending
    # is bimodal like this -- a meaningful chunk of legitimate transactions
    # are large, which is exactly what makes a blanket "flag everything
    # over $X" rule dangerous: it floods reviewers with legitimate large
    # purchases, not just fraud. Ordinary fraud tends to cluster at
    # smaller "testing" amounts. The confidently-missed fraud rows are
    # deliberately high value -- that's what makes them costly when
    # missed, mirroring the specific high-value miss cases described in
    # the write-up this project is based on.
    everyday_mask = rng.random(n_legit) < 0.72
    legit_amount = np.where(
        everyday_mask,
        rng.lognormal(mean=2.9, sigma=0.85, size=n_legit),
        rng.lognormal(mean=6.6, sigma=0.5, size=n_legit),  # big-ticket legit purchases
    )
    fraud_amount = rng.lognormal(mean=2.6, sigma=1.3, size=n_fraud - n_confidently_missed)
    missed_amount = rng.uniform(800, 2500, size=n_confidently_missed)

    X = np.vstack([legit_X, fraud_X, missed_X])
    amount = np.concatenate([legit_amount, fraud_amount, missed_amount])
    y = np.concatenate([
        np.zeros(n_legit, dtype=int),
        np.ones(n_fraud - n_confidently_missed, dtype=int),
        np.ones(n_confidently_missed, dtype=int),
    ])
    is_confidently_missed = np.concatenate([
        np.zeros(n_legit, dtype=int),
        np.zeros(n_fraud - n_confidently_missed, dtype=int),
        np.ones(n_confidently_missed, dtype=int),
    ])

    n_total = n_legit + n_fraud
    time = np.sort(rng.uniform(0, 172_800, size=n_total))  # 48h window, seconds

    cols = {f"V{i+1}": X[:, i] for i in range(N_COMPONENTS)}
    df = pd.DataFrame(cols)
    df.insert(0, "Time", time)
    df["Amount"] = np.round(amount, 2)
    df["Class"] = y
    df["_is_confidently_missed"] = is_confidently_missed  # debug/eval column only

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["Time"] = np.sort(df["Time"].values)  # keep time monotonic after shuffle
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud dataset")
    parser.add_argument("--out", default="data/transactions.csv")
    parser.add_argument("--n-legit", type=int, default=10_000)
    parser.add_argument("--n-fraud", type=int, default=200)
    parser.add_argument("--n-confidently-missed", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.n_legit, args.n_fraud, args.n_confidently_missed, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows ({df['Class'].sum()} fraud) to {args.out}")


if __name__ == "__main__":
    main()
