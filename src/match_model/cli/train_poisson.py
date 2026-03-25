from __future__ import annotations

import argparse
import joblib

from match_model.data.loaders import load_csv, normalize_columns
from match_model.models.poisson_model import PoissonGoalModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument(
        "--output", required=True, help="Path to save trained Poisson bundle"
    )
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--half-life-days", type=float, default=180.0)
    parser.add_argument("--shrinkage-k", type=float, default=12.0)
    args = parser.parse_args()

    raw_df = load_csv(args.input)
    df = normalize_columns(raw_df)

    model = PoissonGoalModel(
        max_goals=args.max_goals,
        half_life_days=args.half_life_days,
        shrinkage_k=args.shrinkage_k,
    )
    model.fit(df)

    bundle = {
        "model_type": "poisson_goal_model_v1",
        "feature_columns": ["home_team", "away_team", "date"],
        "model": model,
    }

    joblib.dump(bundle, args.output)

    print(f"Saved trained Poisson model bundle to {args.output}")
