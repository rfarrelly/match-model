from __future__ import annotations

import argparse

from match_model.data.loaders import load_csv, normalize_columns
from match_model.runtime.build_poisson_artifact import build_poisson_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--league", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regularization-strength", type=float, default=0.3)
    parser.add_argument("--half-life-days", type=float, default=180.0)
    parser.add_argument("--rho", type=float, default=-0.02)

    args = parser.parse_args()

    df = normalize_columns(load_csv(args.input))

    build_poisson_artifact(
        df=df,
        league=args.league,
        target_season=args.season,
        output_path=args.output,
        regularization_strength=args.regularization_strength,
        half_life_days=args.half_life_days,
        rho=args.rho,
    )


if __name__ == "__main__":
    main()
