from __future__ import annotations

import argparse

from match_model.data.loaders import load_csv, normalize_columns
from match_model.models.poisson_model import PoissonGoalModel
from match_model.training.regime import LeagueSeasonRegime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--league", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--output-predictions", default="league_regime_predictions.csv")
    parser.add_argument("--output-metrics", default="league_regime_metrics.csv")
    parser.add_argument("--half-life-days", type=float, default=180.0)
    parser.add_argument("--regularization-strength", type=float, default=1.0)
    parser.add_argument("--use-dixon-coles", action="store_true")
    parser.add_argument("--rho", type=float, default=-0.02)
    args = parser.parse_args()

    df = normalize_columns(load_csv(args.input))

    regime = LeagueSeasonRegime(
        league=args.league,
        target_season=args.season,
        drop_promoted_relegated=True,
    )

    predictions_df, metrics_df = regime.run(
        df=df,
        model_factory=lambda: PoissonGoalModel(
            max_goals=10,
            half_life_days=args.half_life_days,
            regularization_strength=args.regularization_strength,
            use_dixon_coles=args.use_dixon_coles,
            rho=args.rho,
        ),
    )

    predictions_df.to_csv(args.output_predictions, index=False)
    metrics_df.to_csv(args.output_metrics, index=False)

    print(metrics_df)
    print()
    print(f"Saved predictions to {args.output_predictions}")
    print(f"Saved metrics to {args.output_metrics}")


if __name__ == "__main__":
    main()
