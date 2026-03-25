from __future__ import annotations

import argparse
import joblib

from match_model.runtime.artifact import save_artifact
from match_model.runtime.scorer import OutcomeScorerArtifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-model", required=True, help="Path to trained model bundle"
    )
    parser.add_argument(
        "--output-artifact", required=True, help="Path to save scorer artifact"
    )
    args = parser.parse_args()

    bundle = joblib.load(args.input_model)

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    model_type = bundle.get("model_type", "unknown")

    artifact = OutcomeScorerArtifact(
        model=model,
        feature_columns=feature_columns,
        model_name=model_type,
    )

    save_artifact(artifact, args.output_artifact)
    print(f"Saved scorer artifact to {args.output_artifact}")
