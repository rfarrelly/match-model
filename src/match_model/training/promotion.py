from __future__ import annotations

import pandas as pd


def teams_in_season(df: pd.DataFrame) -> set[str]:
    return set(pd.unique(df[["home_team", "away_team"]].values.ravel()))


def continuing_teams(
    previous_season_df: pd.DataFrame,
    current_season_df: pd.DataFrame,
) -> set[str]:
    prev = teams_in_season(previous_season_df)
    curr = teams_in_season(current_season_df)
    return prev & curr


def filter_matches_to_team_pool(
    df: pd.DataFrame,
    allowed_teams: set[str],
) -> pd.DataFrame:
    mask = df["home_team"].isin(allowed_teams) & df["away_team"].isin(allowed_teams)
    return df.loc[mask].copy()
