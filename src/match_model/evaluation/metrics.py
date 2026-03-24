from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


def multiclass_log_loss(y_true: pd.Series, prob_df: pd.DataFrame) -> float:
    aligned = prob_df.rename(
        columns={
            "home_win_prob": "H",
            "draw_prob": "D",
            "away_win_prob": "A",
        }
    )[["A", "D", "H"]]

    return float(log_loss(y_true, aligned, labels=["A", "D", "H"]))


def multiclass_accuracy(y_true: pd.Series, predicted: pd.Series) -> float:
    return float(accuracy_score(y_true, predicted))
