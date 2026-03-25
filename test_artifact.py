import pandas as pd
from match_model.data.loaders import load_csv, normalize_columns
from match_model.runtime.artifact import load_artifact

df = normalize_columns(
    load_csv("/Users/ryanfarrelly/Desktop/feature-engine/OUTPUT/historical_ppi.csv")
)
artifact = load_artifact("artifacts/multiclass_baseline_scorer.joblib")
preds = artifact.score_dataframe(df.head(10))
print(preds)
