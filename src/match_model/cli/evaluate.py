from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from match_model.data.loaders import load_csv, normalize_columns
from match_model.evaluation.walk_forward import evaluate_walk_forward
from match_model.models.multiclass_baseline import MulticlassBaselineModel


DEFAULT_FEATURE_COLUMNS = [
    "odds_home",
    "odds_draw",
    "odds_away",
    "home_ppi",
    "away_ppi",
    "ppi_diff",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--step-size", type=int, default=300)
    parser.add_argument("--output", default="walk_forward_metrics.csv")
    args = parser.parse_args()

    raw_df = load_csv(args.input)
    df = normalize_columns(raw_df)

    feature_columns = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    if not feature_columns:
        raise ValueError("No baseline feature columns found after normalization")

    metrics_df = evaluate_walk_forward(
        df=df,
        model_factory=lambda: MulticlassBaselineModel(feature_columns=feature_columns),
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )

    metrics_df.to_csv(args.output, index=False)

    print("Walk-forward evaluation complete")
    print(metrics_df)
    print()
    print("Feature columns used:", feature_columns)
    print("Average log loss:", metrics_df["log_loss"].mean())
    print("Average accuracy:", metrics_df["accuracy"].mean())


if __name__ == "__main__":
    main()
