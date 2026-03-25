from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import poisson

from match_model.models.base import BaseOutcomeModel


@dataclass
class TeamStrength:
    home_attack: float = 1.0
    away_attack: float = 1.0
    home_defence: float = 1.0
    away_defence: float = 1.0


class PoissonGoalModel(BaseOutcomeModel):
    """
    Production-oriented Poisson goal model with:
    - time decay
    - shrinkage toward league average
    - sane defaults for unseen teams

    Predicts:
        home_win_prob
        draw_prob
        away_win_prob

    Also exposes expected goals through predict_expected_goals().
    """

    def __init__(
        self,
        max_goals: int = 10,
        half_life_days: float = 180.0,
        shrinkage_k: float = 12.0,
        min_lambda: float = 0.05,
        max_lambda: float = 4.5,
    ):
        self.max_goals = max_goals
        self.half_life_days = half_life_days
        self.shrinkage_k = shrinkage_k
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        self.avg_home_goals: float = 1.0
        self.avg_away_goals: float = 1.0
        self.team_strengths: dict[str, TeamStrength] = {}
        self.fitted_: bool = False
        self.training_cutoff_date_: pd.Timestamp | None = None

    def fit(self, df: pd.DataFrame) -> "PoissonGoalModel":
        required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for Poisson fit: {sorted(missing)}"
            )

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="raise")
        data = data.sort_values("date").reset_index(drop=True)

        self.training_cutoff_date_ = data["date"].max()

        # Time-decay weights: more recent matches matter more.
        weights = self._compute_time_decay_weights(
            data["date"], self.training_cutoff_date_
        )
        data["_weight"] = weights

        self.avg_home_goals = self._weighted_mean(data["home_goals"], data["_weight"])
        self.avg_away_goals = self._weighted_mean(data["away_goals"], data["_weight"])

        teams = pd.unique(data[["home_team", "away_team"]].values.ravel())
        self.team_strengths = {}

        for team in teams:
            self.team_strengths[team] = TeamStrength(
                home_attack=self._estimate_home_attack(data, team),
                away_attack=self._estimate_away_attack(data, team),
                home_defence=self._estimate_home_defence(data, team),
                away_defence=self._estimate_away_defence(data, team),
            )

        self.fitted_ = True
        return self

    def predict_expected_goals(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ValueError("Model must be fitted before predict_expected_goals()")

        rows = []
        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            home_strength = self.team_strengths.get(home, TeamStrength())
            away_strength = self.team_strengths.get(away, TeamStrength())

            lambda_home = (
                self.avg_home_goals
                * home_strength.home_attack
                * away_strength.away_defence
            )
            lambda_away = (
                self.avg_away_goals
                * away_strength.away_attack
                * home_strength.home_defence
            )

            lambda_home = float(np.clip(lambda_home, self.min_lambda, self.max_lambda))
            lambda_away = float(np.clip(lambda_away, self.min_lambda, self.max_lambda))

            rows.append(
                {
                    "expected_home_goals": lambda_home,
                    "expected_away_goals": lambda_away,
                }
            )

        return pd.DataFrame(rows, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        xg = self.predict_expected_goals(df)

        rows = []
        for _, row in xg.iterrows():
            probs = self._match_outcome_probs(
                lambda_home=row["expected_home_goals"],
                lambda_away=row["expected_away_goals"],
            )
            rows.append(probs)

        return pd.DataFrame(rows, index=df.index)

    def _match_outcome_probs(
        self, lambda_home: float, lambda_away: float
    ) -> dict[str, float]:
        goals = np.arange(self.max_goals + 1)

        home_probs = poisson.pmf(goals, lambda_home)
        away_probs = poisson.pmf(goals, lambda_away)

        matrix = np.outer(home_probs, away_probs)

        # Rows = home goals, cols = away goals
        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()

        total = home_win + draw + away_win
        if total <= 0:
            return {
                "home_win_prob": 1 / 3,
                "draw_prob": 1 / 3,
                "away_win_prob": 1 / 3,
            }

        return {
            "home_win_prob": float(home_win / total),
            "draw_prob": float(draw / total),
            "away_win_prob": float(away_win / total),
        }

    def _compute_time_decay_weights(
        self,
        dates: pd.Series,
        reference_date: pd.Timestamp,
    ) -> np.ndarray:
        ages_days = (reference_date - pd.to_datetime(dates)).dt.days.clip(lower=0)
        # Half-life decay: weight halves every `half_life_days`
        return np.power(0.5, ages_days / self.half_life_days)

    def _weighted_mean(
        self, values: pd.Series, weights: pd.Series | np.ndarray
    ) -> float:
        values_arr = np.asarray(values, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        denom = weights_arr.sum()
        if denom <= 0:
            return float(np.mean(values_arr))
        return float(np.sum(values_arr * weights_arr) / denom)

    def _shrink_ratio(
        self,
        weighted_mean: float,
        baseline_mean: float,
        effective_n: float,
    ) -> float:
        """
        Shrink team-specific mean toward league baseline mean.
        """
        shrunk_mean = (
            (effective_n * weighted_mean) + (self.shrinkage_k * baseline_mean)
        ) / (effective_n + self.shrinkage_k)

        if baseline_mean <= 0:
            return 1.0
        return float(shrunk_mean / baseline_mean)

    def _effective_sample_size(self, weights: pd.Series) -> float:
        # Simple, intuitive "effective count" as sum of weights.
        return float(weights.sum())

    def _estimate_home_attack(self, data: pd.DataFrame, team: str) -> float:
        games = data[data["home_team"] == team]
        if len(games) == 0:
            return 1.0

        weighted_mean = self._weighted_mean(games["home_goals"], games["_weight"])
        eff_n = self._effective_sample_size(games["_weight"])
        return self._shrink_ratio(weighted_mean, self.avg_home_goals, eff_n)

    def _estimate_away_attack(self, data: pd.DataFrame, team: str) -> float:
        games = data[data["away_team"] == team]
        if len(games) == 0:
            return 1.0

        weighted_mean = self._weighted_mean(games["away_goals"], games["_weight"])
        eff_n = self._effective_sample_size(games["_weight"])
        return self._shrink_ratio(weighted_mean, self.avg_away_goals, eff_n)

    def _estimate_home_defence(self, data: pd.DataFrame, team: str) -> float:
        games = data[data["home_team"] == team]
        if len(games) == 0:
            return 1.0

        weighted_mean = self._weighted_mean(games["away_goals"], games["_weight"])
        eff_n = self._effective_sample_size(games["_weight"])
        return self._shrink_ratio(weighted_mean, self.avg_away_goals, eff_n)

    def _estimate_away_defence(self, data: pd.DataFrame, team: str) -> float:
        games = data[data["away_team"] == team]
        if len(games) == 0:
            return 1.0

        weighted_mean = self._weighted_mean(games["home_goals"], games["_weight"])
        eff_n = self._effective_sample_size(games["_weight"])
        return self._shrink_ratio(weighted_mean, self.avg_home_goals, eff_n)
