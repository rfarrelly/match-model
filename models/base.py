from abc import ABC, abstractmethod
import pandas as pd


class BaseOutcomeModel(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseOutcomeModel":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe with the same index as input and columns:
        - home_win_prob
        - draw_prob
        - away_win_prob
        """
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> pd.Series:
        probs = self.predict_proba(df)
        return probs.idxmax(axis=1).map(
            {
                "home_win_prob": "H",
                "draw_prob": "D",
                "away_win_prob": "A",
            }
        )
