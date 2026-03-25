from __future__ import annotations

import pandas as pd


class OutcomeScorerArtifact:
    def __init__(self, model, feature_columns: list[str], model_name: str = "unknown"):
        self.model = model
        self.feature_columns = feature_columns
        self.model_name = model_name

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = self.model.predict_proba(df)

        if not {"home_win_prob", "draw_prob", "away_win_prob"} <= set(probs.columns):
            raise ValueError(
                "Model predict_proba() must return columns: "
                "home_win_prob, draw_prob, away_win_prob"
            )

        out = probs.copy()
        out["predicted_outcome"] = (
            out[["home_win_prob", "draw_prob", "away_win_prob"]]
            .idxmax(axis=1)
            .map(
                {
                    "home_win_prob": "H",
                    "draw_prob": "D",
                    "away_win_prob": "A",
                }
            )
        )

        out["model_name"] = self.model_name
        return out
