REQUIRED_CORE_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "result",
]

REQUIRED_TRAINING_COLUMNS = [
    "home_goals",
    "away_goals",
]

ODDS_COLUMNS = [
    "odds_home",
    "odds_draw",
    "odds_away",
]

OPTIONAL_FEATURE_COLUMNS = [
    "home_ppi",
    "away_ppi",
    "ppi_diff",
]

CANONICAL_RESULT_VALUES = {"H", "D", "A"}
