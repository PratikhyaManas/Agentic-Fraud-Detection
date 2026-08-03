from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.decision import Action
from src.runtime import build_agent, load_model_and_decision_layer, maybe_load_dotenv
from src.summarize import Summarizer

MODEL_PATH = Path("models/fraud_model.pkl")
CONFIG_PATH = Path("models/decision_config.json")


def _timed_run(agent, df: pd.DataFrame, top_k_features: int) -> dict:
    start = time.perf_counter()
    outcomes = agent.run(df, top_k_features=top_k_features)
    elapsed = time.perf_counter() - start
    n = len(outcomes)
    return {
        "rows": n,
        "seconds": elapsed,
        "rows_per_second": (n / elapsed) if elapsed > 0 else float("inf"),
    }


def main() -> None:
    maybe_load_dotenv()

    parser = argparse.ArgumentParser(description="Benchmark scoring throughput with and without summaries")
    parser.add_argument("--input", default="data/transactions.csv", help="Input CSV")
    parser.add_argument("--rows", type=int, default=2000, help="Number of rows to benchmark")
    parser.add_argument("--top-k-features", type=int, default=3, help="Top semantic signals per explained row")
    parser.add_argument("--output", default="outputs/benchmark.json", help="Path to write JSON benchmark results")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run `python train.py` first.")

    df = pd.read_csv(args.input).head(args.rows)
    model, layer, _ = load_model_and_decision_layer(MODEL_PATH, CONFIG_PATH)

    fast_agent = build_agent(
        model=model,
        decision_layer=layer,
        background=None,
        summarizer=None,
        explain_actions=(),
    )
    full_agent = build_agent(
        model=model,
        decision_layer=layer,
        background=df,
        summarizer=Summarizer(),
        explain_actions=(Action.REVIEW, Action.BLOCK),
    )

    fast = _timed_run(fast_agent, df, args.top_k_features)
    full = _timed_run(full_agent, df, args.top_k_features)

    result = {
        "input": args.input,
        "rows": int(len(df)),
        "top_k_features": args.top_k_features,
        "fast_no_summaries": fast,
        "full_with_summaries": full,
        "speedup_no_summaries_vs_full": (
            fast["rows_per_second"] / full["rows_per_second"] if full["rows_per_second"] > 0 else None
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print("Benchmark complete")
    print(f"  rows: {result['rows']:,}")
    print(f"  fast mode: {fast['seconds']:.3f}s ({fast['rows_per_second']:.1f} rows/s)")
    print(f"  full mode: {full['seconds']:.3f}s ({full['rows_per_second']:.1f} rows/s)")
    print(f"  speedup: {result['speedup_no_summaries_vs_full']:.2f}x")
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()