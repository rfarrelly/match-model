import pandas as pd

from data.splits import generate_walk_forward_folds
from evaluation.metrics import multiclass_accuracy, multiclass_log_loss


def evaluate_walk_forward(
    df: pd.DataFrame,
    model_factory,
    train_size: int,
    test_size: int,
    step_size: int,
) -> pd.DataFrame:
    rows = []

    folds = generate_walk_forward_folds(
        df=df,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )

    for fold in folds:
        model = model_factory()
        model.fit(fold.train_df)

        prob_df = model.predict_proba(fold.test_df)
        pred = model.predict(fold.test_df)

        rows.append(
            {
                "fold_index": fold.fold_index,
                "train_rows": len(fold.train_df),
                "test_rows": len(fold.test_df),
                "log_loss": multiclass_log_loss(fold.test_df["result"], prob_df),
                "accuracy": multiclass_accuracy(fold.test_df["result"], pred),
                "test_start_date": fold.test_df["date"].min(),
                "test_end_date": fold.test_df["date"].max(),
            }
        )

    return pd.DataFrame(rows)
