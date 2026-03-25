from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class MulticlassProbabilityCalibrator:
    """
    Simple multiclass calibrator:
    trains one multinomial logistic regression on raw probability triplets.
    """

    def __init__(self):
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )
        self.fitted_ = False

    def fit(
        self, prob_df: pd.DataFrame, y: pd.Series
    ) -> "MulticlassProbabilityCalibrator":
        X = prob_df[["home_win_prob", "draw_prob", "away_win_prob"]].to_numpy()
        self.model.fit(X, y)
        self.fitted_ = True
        return self

    def transform(self, prob_df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ValueError("Calibrator must be fitted before transform()")

        X = prob_df[["home_win_prob", "draw_prob", "away_win_prob"]].to_numpy()
        probs = self.model.predict_proba(X)

        class_order = list(self.model.classes_)
        tmp = pd.DataFrame(probs, index=prob_df.index, columns=class_order)

        return pd.DataFrame(
            {
                "home_win_prob": tmp["H"] if "H" in tmp else 0.0,
                "draw_prob": tmp["D"] if "D" in tmp else 0.0,
                "away_win_prob": tmp["A"] if "A" in tmp else 0.0,
            },
            index=prob_df.index,
        )
