from __future__ import annotations

import pandas as pd

from match_model.data.splits import generate_walk_forward_folds
from match_model.evaluation.metrics import multiclass_accuracy, multiclass_log_loss


def evaluate_walk_forward(
    df: pd.DataFrame,
    model_factory,
    train_size: int,
    test_size: int,
    step_size: int,
    calibrator_factory=None,
) -> pd.DataFrame:
    rows: list[dict] = []

    folds = generate_walk_forward_folds(
        df=df,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )

    for fold in folds:
        model = model_factory()
        model.fit(fold.train_df)

        train_prob_df = model.predict_proba(fold.train_df)
        test_prob_df = model.predict_proba(fold.test_df)

        if calibrator_factory is not None:
            calibrator = calibrator_factory()
            calibrator.fit(train_prob_df, fold.train_df["result"])
            test_prob_df = calibrator.transform(test_prob_df)

        pred = test_prob_df.idxmax(axis=1).map(
            {
                "home_win_prob": "H",
                "draw_prob": "D",
                "away_win_prob": "A",
            }
        )

        rows.append(
            {
                "fold_index": fold.fold_index,
                "train_rows": len(fold.train_df),
                "test_rows": len(fold.test_df),
                "log_loss": multiclass_log_loss(fold.test_df["result"], test_prob_df),
                "accuracy": multiclass_accuracy(fold.test_df["result"], pred),
                "test_start_date": fold.test_df["date"].min(),
                "test_end_date": fold.test_df["date"].max(),
            }
        )

    return pd.DataFrame(rows)
