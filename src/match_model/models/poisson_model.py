from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from match_model.models.base import BaseOutcomeModel


class PoissonGoalModel(BaseOutcomeModel):
    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals

        self.home_attack: dict[str, float] = {}
        self.away_attack: dict[str, float] = {}
        self.home_defence: dict[str, float] = {}
        self.away_defence: dict[str, float] = {}

        self.avg_home_goals: float = 1.0
        self.avg_away_goals: float = 1.0

    def fit(self, df: pd.DataFrame) -> "PoissonGoalModel":
        # league averages
        self.avg_home_goals = df["home_goals"].mean()
        self.avg_away_goals = df["away_goals"].mean()

        teams = pd.unique(df[["home_team", "away_team"]].values.ravel())

        # initialize
        for t in teams:
            self.home_attack[t] = 1.0
            self.away_attack[t] = 1.0
            self.home_defence[t] = 1.0
            self.away_defence[t] = 1.0

        # simple ratio estimation (can be improved later)
        for t in teams:
            home_games = df[df["home_team"] == t]
            away_games = df[df["away_team"] == t]

            if len(home_games) > 0:
                self.home_attack[t] = (
                    home_games["home_goals"].mean() / self.avg_home_goals
                )
                self.home_defence[t] = (
                    home_games["away_goals"].mean() / self.avg_away_goals
                )

            if len(away_games) > 0:
                self.away_attack[t] = (
                    away_games["away_goals"].mean() / self.avg_away_goals
                )
                self.away_defence[t] = (
                    away_games["home_goals"].mean() / self.avg_home_goals
                )

        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            λ_home = (
                self.avg_home_goals
                * self.home_attack.get(home, 1.0)
                * self.away_defence.get(away, 1.0)
            )

            λ_away = (
                self.avg_away_goals
                * self.away_attack.get(away, 1.0)
                * self.home_defence.get(home, 1.0)
            )

            probs = self._match_outcome_probs(λ_home, λ_away)

            rows.append(probs)

        return pd.DataFrame(rows, index=df.index)

    def _match_outcome_probs(self, λ_home: float, λ_away: float) -> dict:
        max_g = self.max_goals

        home_goals_probs = poisson.pmf(np.arange(max_g + 1), λ_home)
        away_goals_probs = poisson.pmf(np.arange(max_g + 1), λ_away)

        matrix = np.outer(home_goals_probs, away_goals_probs)

        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()

        total = home_win + draw + away_win

        return {
            "home_win_prob": home_win / total,
            "draw_prob": draw / total,
            "away_win_prob": away_win / total,
        }
