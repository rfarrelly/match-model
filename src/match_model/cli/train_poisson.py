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
    parser.add_argument("--regularization-strength", type=float, default=1.0)
    parser.add_argument("--use-dixon-coles", action="store_true")
    parser.add_argument("--rho", type=float, default=-0.02)
    args = parser.parse_args()

    raw_df = load_csv(args.input)
    df = normalize_columns(raw_df)

    model = PoissonGoalModel(
        max_goals=args.max_goals,
        half_life_days=args.half_life_days,
        regularization_strength=args.regularization_strength,
        use_dixon_coles=args.use_dixon_coles,
        rho=args.rho,
    )
    model.fit(df)

    bundle = {
        "model_type": "poisson_goal_model_mle_v1",
        "feature_columns": ["home_team", "away_team", "date"],
        "model": model,
    }

    joblib.dump(bundle, args.output)

    print(f"Saved trained Poisson model bundle to {args.output}")
