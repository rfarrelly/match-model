from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class BaseOutcomeModel(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseOutcomeModel":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Must return:
            home_win_prob
            draw_prob
            away_win_prob
        """
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> pd.Series:
        probs = self.predict_proba(df)

        return (
            probs[["home_win_prob", "draw_prob", "away_win_prob"]]
            .idxmax(axis=1)
            .map(
                {
                    "home_win_prob": "H",
                    "draw_prob": "D",
                    "away_win_prob": "A",
                }
            )
        )
