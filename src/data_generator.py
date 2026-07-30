"""
Synthetic credit-card transaction generator for fraud detection demos.
Produces realistic-ish features and a minority fraud class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


FEATURE_NAMES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "distance_from_home_km",
    "distance_from_last_txn_km",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]


def generate_transactions(
    n_samples: int = 50_000,
    fraud_rate: float = 0.017,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset roughly inspired by public
    credit-card fraud benchmarks.

    Returns
    -------
    DataFrame with columns FEATURE_NAMES + ['is_fraud']
    """
    rng = np.random.default_rng(random_state)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # ---- Legitimate transactions ----
    legit = {
        "amount": rng.lognormal(mean=3.5, sigma=1.1, size=n_legit).clip(1, 5_000),
        "hour_of_day": rng.integers(0, 24, size=n_legit),
        "day_of_week": rng.integers(0, 7, size=n_legit),
        "distance_from_home_km": rng.exponential(scale=15, size=n_legit).clip(0, 300),
        "distance_from_last_txn_km": rng.exponential(scale=8, size=n_legit).clip(0, 200),
        "ratio_to_median_purchase_price": rng.lognormal(0, 0.4, size=n_legit).clip(0.1, 8),
        "repeat_retailer": rng.binomial(1, 0.75, size=n_legit),
        "used_chip": rng.binomial(1, 0.85, size=n_legit),
        "used_pin_number": rng.binomial(1, 0.35, size=n_legit),
        "online_order": rng.binomial(1, 0.40, size=n_legit),
    }

    # ---- Fraudulent transactions (skewed toward high-risk patterns) ----
    fraud = {
        "amount": rng.lognormal(mean=5.2, sigma=1.3, size=n_fraud).clip(5, 20_000),
        "hour_of_day": rng.choice(
            [0, 1, 2, 3, 4, 22, 23], size=n_fraud, p=[0.15, 0.15, 0.15, 0.1, 0.1, 0.15, 0.2]
        ),
        "day_of_week": rng.integers(0, 7, size=n_fraud),
        "distance_from_home_km": rng.exponential(scale=80, size=n_fraud).clip(0, 2_000),
        "distance_from_last_txn_km": rng.exponential(scale=120, size=n_fraud).clip(0, 3_000),
        "ratio_to_median_purchase_price": rng.lognormal(1.2, 0.8, size=n_fraud).clip(0.5, 30),
        "repeat_retailer": rng.binomial(1, 0.25, size=n_fraud),
        "used_chip": rng.binomial(1, 0.30, size=n_fraud),
        "used_pin_number": rng.binomial(1, 0.08, size=n_fraud),
        "online_order": rng.binomial(1, 0.85, size=n_fraud),
    }

    df_legit = pd.DataFrame(legit)
    df_legit["is_fraud"] = 0
    df_fraud = pd.DataFrame(fraud)
    df_fraud["is_fraud"] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Add a transaction id for traceability
    df.insert(0, "transaction_id", [f"TXN-{i:08d}" for i in range(len(df))])

    return df


def train_test_split_df(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple stratified split on is_fraud."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["is_fraud"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    df = generate_transactions(5_000)
    print(df.head())
    print("\nClass balance:")
    print(df["is_fraud"].value_counts(normalize=True))
    print(f"\nShape: {df.shape}")