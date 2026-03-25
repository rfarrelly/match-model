from __future__ import annotations

import argparse

from match_model.data.loaders import load_csv, normalize_columns
from match_model.evaluation.calibration import MulticlassProbabilityCalibrator
from match_model.evaluation.walk_forward import evaluate_walk_forward
from match_model.models.poisson_model import PoissonGoalModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--step-size", type=int, default=300)
    parser.add_argument(
        "--output", default="poisson_calibrated_walk_forward_metrics.csv"
    )
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--half-life-days", type=float, default=180.0)
    parser.add_argument("--shrinkage-k", type=float, default=12.0)
    args = parser.parse_args()

    df = normalize_columns(load_csv(args.input))

    metrics = evaluate_walk_forward(
        df=df,
        model_factory=lambda: PoissonGoalModel(
            max_goals=args.max_goals,
            half_life_days=args.half_life_days,
            shrinkage_k=args.shrinkage_k,
        ),
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        calibrator_factory=MulticlassProbabilityCalibrator,
    )

    metrics.to_csv(args.output, index=False)

    print(metrics)
    print()
    print("Average log loss:", metrics["log_loss"].mean())
    print("Average accuracy:", metrics["accuracy"].mean())
