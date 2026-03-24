import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from models.base import BaseOutcomeModel


class MulticlassBaselineModel(BaseOutcomeModel):
    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        multi_class="multinomial",
                        max_iter=1000,
                    ),
                ),
            ]
        )
        self.class_order_: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> "MulticlassBaselineModel":
        X = df[self.feature_columns]
        y = df["result"]

        self.pipeline.fit(X, y)
        self.class_order_ = list(self.pipeline.named_steps["model"].classes_)
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.class_order_ is None:
            raise ValueError("Model must be fitted before predict_proba is called")

        X = df[self.feature_columns]
        probs = self.pipeline.predict_proba(X)

        proba_df = pd.DataFrame(
            probs,
            index=df.index,
            columns=self.class_order_,
        )

        return pd.DataFrame(
            {
                "home_win_prob": proba_df.get("H", pd.Series(0.0, index=df.index)),
                "draw_prob": proba_df.get("D", pd.Series(0.0, index=df.index)),
                "away_win_prob": proba_df.get("A", pd.Series(0.0, index=df.index)),
            },
            index=df.index,
        )
