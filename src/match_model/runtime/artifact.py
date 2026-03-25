from __future__ import annotations

from pathlib import Path
import joblib

from match_model.runtime.scorer import OutcomeScorerArtifact


def save_artifact(
    artifact: OutcomeScorerArtifact,
    path: str | Path,
) -> None:
    joblib.dump(artifact, path)


def load_artifact(path: str | Path) -> OutcomeScorerArtifact:
    obj = joblib.load(path)
    if not hasattr(obj, "score_dataframe"):
        raise ValueError("Loaded object does not implement score_dataframe(df)")
    return obj
