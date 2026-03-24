import pandas as pd

from schema.columns import (
    CANONICAL_RESULT_VALUES,
    REQUIRED_CORE_COLUMNS,
)


def validate_core_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_CORE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    invalid = set(df["result"].dropna().unique()) - CANONICAL_RESULT_VALUES
    if invalid:
        raise ValueError(f"Invalid result values found: {sorted(invalid)}")
