from __future__ import annotations

import argparse

from match_model.data.loaders import load_csv, normalize_columns
from match_model.evaluation.walk_forward import evaluate_walk_forward
from match_model.models.poisson_model import PoissonGoalModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--step-size", type=int, default=300)
    args = parser.parse_args()

    df = normalize_columns(load_csv(args.input))

    metrics = evaluate_walk_forward(
        df=df,
        model_factory=lambda: PoissonGoalModel(),
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )

    print(metrics)
    print("Avg log loss:", metrics["log_loss"].mean())
