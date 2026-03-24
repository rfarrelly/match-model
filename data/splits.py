from dataclasses import dataclass
import pandas as pd


@dataclass
class WalkForwardFold:
    fold_index: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def generate_walk_forward_folds(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int,
) -> list[WalkForwardFold]:
    folds: list[WalkForwardFold] = []
    start = 0
    fold_index = 0

    while start + train_size + test_size <= len(df):
        train_df = df.iloc[start : start + train_size].copy()
        test_df = df.iloc[start + train_size : start + train_size + test_size].copy()

        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                train_df=train_df,
                test_df=test_df,
            )
        )

        fold_index += 1
        start += step_size

    return folds
