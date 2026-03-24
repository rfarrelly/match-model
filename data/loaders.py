import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="raise")
    return out


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("date").reset_index(drop=True)
