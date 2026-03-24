from __future__ import annotations

import pandas as pd

from match_model.schema.columns import CANONICAL_RESULT_VALUES, REQUIRED_CORE_COLUMNS


def validate_core_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_CORE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["date"].isna().any():
        raise ValueError("Column 'date' contains nulls")

    if df["home_team"].isna().any() or df["away_team"].isna().any():
        raise ValueError("Team columns contain nulls")

    invalid = set(df["result"].dropna().unique()) - CANONICAL_RESULT_VALUES
    if invalid:
        raise ValueError(f"Invalid result values found: {sorted(invalid)}")
