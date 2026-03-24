from __future__ import annotations

import argparse
import joblib

from match_model.data.loaders import load_csv, normalize_columns
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
    parser.add_argument("--output", required=True, help="Path to save trained baseline")
    args = parser.parse_args()

    raw_df = load_csv(args.input)
    df = normalize_columns(raw_df)

    feature_columns = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    if not feature_columns:
        raise ValueError("No baseline feature columns found after normalization")

    model = MulticlassBaselineModel(feature_columns=feature_columns)
    model.fit(df)

    joblib.dump(
        {
            "model_type": "multiclass_baseline_v1",
            "feature_columns": feature_columns,
            "model": model,
        },
        args.output,
    )

    print(f"Saved model to {args.output}")
    print("Feature columns used:", feature_columns)


if __name__ == "__main__":
    main()
