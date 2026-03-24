import pandas as pd
from sklearn.metrics import log_loss, accuracy_score


def multiclass_log_loss(y_true: pd.Series, prob_df: pd.DataFrame) -> float:
    return float(
        log_loss(
            y_true,
            prob_df[["home_win_prob", "draw_prob", "away_win_prob"]].rename(
                columns={
                    "home_win_prob": "H",
                    "draw_prob": "D",
                    "away_win_prob": "A",
                }
            ),
            labels=["H", "D", "A"],
        )
    )


def multiclass_accuracy(y_true: pd.Series, predicted: pd.Series) -> float:
    return float(accuracy_score(y_true, predicted))
