from __future__ import annotations

from pathlib import Path
import pandas as pd

from match_model.schema.validation import validate_core_schema


DEFAULT_COLUMN_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
    "PSH": "odds_home",
    "PSD": "odds_draw",
    "PSA": "odds_away",
    "hPPI": "home_ppi",
    "aPPI": "away_ppi",
    "PPIDiff": "ppi_diff",
}


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def normalize_columns(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    mapping = dict(DEFAULT_COLUMN_MAP)
    if column_map:
        mapping.update(column_map)

    renamed = df.rename(
        columns={k: v for k, v in mapping.items() if k in df.columns}
    ).copy()

    if "result" not in renamed.columns and {"home_goals", "away_goals"} <= set(
        renamed.columns
    ):
        renamed["result"] = renamed.apply(_derive_result, axis=1)

    if "ppi_diff" not in renamed.columns and {"home_ppi", "away_ppi"} <= set(
        renamed.columns
    ):
        renamed["ppi_diff"] = renamed["home_ppi"] - renamed["away_ppi"]

    if "date" in renamed.columns:
        renamed["date"] = pd.to_datetime(renamed["date"], errors="raise")

    renamed = renamed.sort_values("date").reset_index(drop=True)
    validate_core_schema(renamed)
    return renamed


def _derive_result(row: pd.Series) -> str:
    if row["home_goals"] > row["away_goals"]:
        return "H"
    if row["home_goals"] < row["away_goals"]:
        return "A"
    return "D"
