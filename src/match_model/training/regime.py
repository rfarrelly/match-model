from __future__ import annotations

import pandas as pd

from match_model.evaluation.metrics import multiclass_accuracy, multiclass_log_loss
from match_model.training.promotion import continuing_teams, filter_matches_to_team_pool
from match_model.training.season_utils import previous_season


class LeagueSeasonRegime:
    """
    Evaluate a model in the style of:
    - train initial model on previous season
    - same league only
    - optionally ignore promoted/relegated teams in carryover initialization
    - iterate through current season chronologically
    - predict before each date block
    - update model after results become known
    """

    def __init__(
        self,
        league: str,
        target_season: str,
        drop_promoted_relegated: bool = True,
    ):
        self.league = league
        self.target_season = target_season
        self.previous_season = previous_season(target_season)
        self.drop_promoted_relegated = drop_promoted_relegated

    def run(
        self,
        df: pd.DataFrame,
        model_factory,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        required = {
            "date",
            "league",
            "season",
            "home_team",
            "away_team",
            "result",
            "home_goals",
            "away_goals",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for regime run: {sorted(missing)}"
            )

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="raise")

        league_df = data[data["league"] == self.league].copy()
        if league_df.empty:
            raise ValueError(f"No rows found for league={self.league}")

        prev_df = league_df[league_df["season"] == self.previous_season].copy()
        curr_df = league_df[league_df["season"] == self.target_season].copy()

        if prev_df.empty:
            raise ValueError(
                f"No previous-season rows found for league={self.league}, season={self.previous_season}"
            )
        if curr_df.empty:
            raise ValueError(
                f"No target-season rows found for league={self.league}, season={self.target_season}"
            )

        prev_df = prev_df.sort_values("date").reset_index(drop=True)
        curr_df = curr_df.sort_values("date").reset_index(drop=True)

        if self.drop_promoted_relegated:
            keep_teams = continuing_teams(prev_df, curr_df)
            prev_df = filter_matches_to_team_pool(prev_df, keep_teams)

        # Rolling training data starts with filtered previous season
        rolling_train_df = prev_df.copy()

        prediction_rows: list[pd.DataFrame] = []

        # Predict in chronological date blocks
        for match_date, block_df in curr_df.groupby("date", sort=True):
            model = model_factory()
            model.fit(rolling_train_df)

            prob_df = model.predict_proba(block_df)
            pred_df = block_df.copy()

            pred_df["home_win_prob"] = prob_df["home_win_prob"].values
            pred_df["draw_prob"] = prob_df["draw_prob"].values
            pred_df["away_win_prob"] = prob_df["away_win_prob"].values

            pred_df["predicted_outcome"] = (
                prob_df[["home_win_prob", "draw_prob", "away_win_prob"]]
                .idxmax(axis=1)
                .map(
                    {
                        "home_win_prob": "H",
                        "draw_prob": "D",
                        "away_win_prob": "A",
                    }
                )
            )

            prediction_rows.append(pred_df)

            # After predicting this block, add the actual played matches into training
            rolling_train_df = (
                pd.concat([rolling_train_df, block_df], ignore_index=True)
                .sort_values("date")
                .reset_index(drop=True)
            )

        predictions_df = pd.concat(prediction_rows, ignore_index=True)

        metrics_df = self._summarize_predictions(predictions_df)
        return predictions_df, metrics_df

    def _summarize_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        log_loss_value = multiclass_log_loss(
            predictions_df["result"],
            predictions_df[["home_win_prob", "draw_prob", "away_win_prob"]],
        )
        accuracy_value = multiclass_accuracy(
            predictions_df["result"],
            predictions_df["predicted_outcome"],
        )

        return pd.DataFrame(
            [
                {
                    "league": self.league,
                    "target_season": self.target_season,
                    "previous_season": self.previous_season,
                    "matches_predicted": len(predictions_df),
                    "log_loss": log_loss_value,
                    "accuracy": accuracy_value,
                }
            ]
        )
