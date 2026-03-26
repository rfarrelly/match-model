from __future__ import annotations

import pandas as pd


class OutcomeScorerArtifact:
    def __init__(self, model, model_name: str = "unknown"):
        self.model = model
        self.model_name = model_name

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = self.model.predict_proba(df)
        xg = self.model.predict_expected_goals(df)

        out = probs.copy()

        out["expected_home_goals"] = xg["expected_home_goals"]
        out["expected_away_goals"] = xg["expected_away_goals"]

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
