from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from match_model.models.base import BaseOutcomeModel


@dataclass
class TeamStrength:
    attack: float = 0.0
    defence: float = 0.0


class PoissonGoalModel(BaseOutcomeModel):
    """
    Poisson football model with:
    - maximum likelihood estimation (MLE)
    - time decay
    - L2 regularization
    - optional Dixon-Coles low-score adjustment at inference

    Model:
        log(lambda_home) = home_advantage + attack[home] - defence[away]
        log(lambda_away) = attack[away] - defence[home]

    Outputs:
        home_win_prob
        draw_prob
        away_win_prob
    """

    def __init__(
        self,
        max_goals: int = 10,
        half_life_days: float = 180.0,
        regularization_strength: float = 1.0,
        min_lambda: float = 0.05,
        max_lambda: float = 4.5,
        use_dixon_coles: bool = False,
        rho: float = -0.02,
        optimizer_maxiter: int = 1000,
        optimizer_maxfun: int = 50000,
    ):
        self.max_goals = max_goals
        self.half_life_days = half_life_days
        self.regularization_strength = regularization_strength
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        self.use_dixon_coles = use_dixon_coles
        self.rho = rho

        self.optimizer_maxiter = optimizer_maxiter
        self.optimizer_maxfun = optimizer_maxfun

        self.teams_: list[str] = []
        self.team_to_idx_: dict[str, int] = {}
        self.team_strengths: dict[str, TeamStrength] = {}

        self.home_advantage_: float = 0.0
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

        weights = self._compute_time_decay_weights(
            data["date"], self.training_cutoff_date_
        )
        data["_weight"] = weights

        self.teams_ = sorted(
            pd.unique(data[["home_team", "away_team"]].values.ravel()).tolist()
        )
        self.team_to_idx_ = {team: idx for idx, team in enumerate(self.teams_)}
        n_teams = len(self.teams_)

        home_idx = data["home_team"].map(self.team_to_idx_).to_numpy(dtype=int)
        away_idx = data["away_team"].map(self.team_to_idx_).to_numpy(dtype=int)
        home_goals = data["home_goals"].to_numpy(dtype=float)
        away_goals = data["away_goals"].to_numpy(dtype=float)
        match_weights = data["_weight"].to_numpy(dtype=float)

        # Parameter vector:
        # [home_advantage, attack_0..attack_n-1, defence_0..defence_n-1]
        x0 = np.zeros(1 + 2 * n_teams, dtype=float)

        avg_home_goals = max(float(data["home_goals"].mean()), 1e-6)
        avg_away_goals = max(float(data["away_goals"].mean()), 1e-6)

        x0[0] = np.log(avg_home_goals / avg_away_goals)

        result = minimize(
            fun=self._objective,
            x0=x0,
            args=(home_idx, away_idx, home_goals, away_goals, match_weights, n_teams),
            method="L-BFGS-B",
            options={
                "maxiter": self.optimizer_maxiter,
                "maxfun": self.optimizer_maxfun,
            },
        )

        if not result.success:
            raise RuntimeError(f"Poisson MLE optimization failed: {result.message}")

        params = result.x
        self.home_advantage_ = float(params[0])

        raw_attack = params[1 : 1 + n_teams]
        raw_defence = params[1 + n_teams : 1 + 2 * n_teams]

        # Center for identifiability / stability
        attack = raw_attack - raw_attack.mean()
        defence = raw_defence - raw_defence.mean()

        self.team_strengths = {
            team: TeamStrength(
                attack=float(attack[idx]),
                defence=float(defence[idx]),
            )
            for team, idx in self.team_to_idx_.items()
        }

        self.fitted_ = True
        return self

    def predict_expected_goals(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ValueError("Model must be fitted before predict_expected_goals()")

        rows = []
        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            home_strength = self.team_strengths.get(home)
            away_strength = self.team_strengths.get(away)

            if home_strength is None:
                home_strength = TeamStrength(attack=-0.05, defence=0.05)
            if away_strength is None:
                away_strength = TeamStrength(attack=-0.05, defence=0.05)

            lambda_home = np.exp(
                self.home_advantage_ + home_strength.attack - away_strength.defence
            )
            lambda_away = np.exp(away_strength.attack - home_strength.defence)

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

    def _objective(
        self,
        params: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
        n_teams: int,
    ) -> float:
        home_advantage = params[0]
        attack = params[1 : 1 + n_teams]
        defence = params[1 + n_teams : 1 + 2 * n_teams]

        # Center attack and defence inside objective for stability / identifiability
        attack = attack - attack.mean()
        defence = defence - defence.mean()

        log_lambda_home = home_advantage + attack[home_idx] - defence[away_idx]
        log_lambda_away = attack[away_idx] - defence[home_idx]

        lambda_home = np.exp(log_lambda_home)
        lambda_away = np.exp(log_lambda_away)

        # Weighted negative log-likelihood (dropping constant factorial terms is fine,
        # but scipy Poisson logpmf is clean and safe)
        ll_home = poisson.logpmf(home_goals, lambda_home)
        ll_away = poisson.logpmf(away_goals, lambda_away)

        weighted_nll = -np.sum(weights * (ll_home + ll_away))

        # L2 regularization on team parameters only
        reg = self.regularization_strength * (np.sum(attack**2) + np.sum(defence**2))

        return float(weighted_nll + reg)

    def _match_outcome_probs(
        self, lambda_home: float, lambda_away: float
    ) -> dict[str, float]:
        goals = np.arange(self.max_goals + 1)

        home_probs = poisson.pmf(goals, lambda_home)
        away_probs = poisson.pmf(goals, lambda_away)

        matrix = np.outer(home_probs, away_probs)

        if self.use_dixon_coles:
            matrix = self._apply_dixon_coles_adjustment(
                matrix, lambda_home, lambda_away
            )

        total = matrix.sum()
        if total <= 0:
            return {
                "home_win_prob": 1 / 3,
                "draw_prob": 1 / 3,
                "away_win_prob": 1 / 3,
            }

        matrix = matrix / total

        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()

        return {
            "home_win_prob": float(home_win),
            "draw_prob": float(draw),
            "away_win_prob": float(away_win),
        }

    def _apply_dixon_coles_adjustment(
        self,
        matrix: np.ndarray,
        lambda_home: float,
        lambda_away: float,
    ) -> np.ndarray:
        adjusted = matrix.copy()
        rho = self.rho

        tau_00 = 1 - (lambda_home * lambda_away * rho)
        tau_01 = 1 + (lambda_home * rho)
        tau_10 = 1 + (lambda_away * rho)
        tau_11 = 1 - rho

        adjusted[0, 0] *= tau_00
        if adjusted.shape[0] > 1:
            adjusted[1, 0] *= tau_10
        if adjusted.shape[1] > 1:
            adjusted[0, 1] *= tau_01
        if adjusted.shape[0] > 1 and adjusted.shape[1] > 1:
            adjusted[1, 1] *= tau_11

        adjusted = np.clip(adjusted, 0.0, None)
        return adjusted

    def _compute_time_decay_weights(
        self,
        dates: pd.Series,
        reference_date: pd.Timestamp,
    ) -> np.ndarray:
        ages_days = (reference_date - pd.to_datetime(dates)).dt.days.clip(lower=0)
        return np.power(0.5, ages_days / self.half_life_days)
