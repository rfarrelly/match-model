from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from match_model.data.loaders import load_csv, normalize_columns
from match_model.models.poisson_model import PoissonGoalModel
from match_model.training.regime import LeagueSeasonRegime


def build_poisson_artifact(
    df: pd.DataFrame,
    league: str,
    target_season: str,
    output_path: str | Path,
    regularization_strength: float = 0.3,
    half_life_days: float = 180.0,
    rho: float = -0.02,
):
    """
    Trains model using league-season regime,
    then saves FINAL fitted model as scorer artifact.
    """

    regime = LeagueSeasonRegime(
        league=league,
        target_season=target_season,
        drop_promoted_relegated=True,
    )

    # Run regime to build rolling training set
    predictions_df, _ = regime.run(
        df=df,
        model_factory=lambda: PoissonGoalModel(
            half_life_days=half_life_days,
            regularization_strength=regularization_strength,
            use_dixon_coles=True,
            rho=rho,
        ),
    )

    # 🔑 Rebuild final model using ALL available training data up to end of season
    league_df = df[df["league"] == league].copy()
    league_df = league_df.sort_values("date")

    final_train_df = league_df[
        league_df["season"].isin([regime.previous_season, target_season])
    ]

    model = PoissonGoalModel(
        half_life_days=half_life_days,
        regularization_strength=regularization_strength,
        use_dixon_coles=True,
        rho=rho,
    )
    model.fit(final_train_df)

    artifact = {
        "model_type": "poisson_mle_regime_v1",
        "league": league,
        "season": target_season,
        "model": model,
    }

    joblib.dump(artifact, output_path)

    print(f"Saved artifact → {output_path}")
