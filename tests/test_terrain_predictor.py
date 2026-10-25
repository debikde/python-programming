"""Check the CSV interface and the local saved validation artifact."""

import csv
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from research_graph.models.terrain_predictor import (
    SOURCE_CHANNELS,
    TerrainPredictor,
    stride_features,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT.parent
    / "research-publication-data"
    / "artifacts"
    / "reports"
    / "model_validation"
)


class ConstantPipeline:
    classes_ = np.array([1, 2])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.ones(len(features), dtype=int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile([0.8, 0.2], (len(features), 1))


def sample_recording() -> pd.DataFrame:
    samples = 70
    frame = pd.DataFrame(
        {
            name: np.linspace(index + 1, index + 2, samples)
            for index, name in enumerate(SOURCE_CHANNELS)
        }
    )
    frame["Steps"] = np.repeat([1, 2], 35)
    frame["Terrain"] = np.repeat([1, 2], 35)
    return frame


def test_csv_prediction_needs_no_terrain_or_turn(tmp_path: Path) -> None:
    frame = sample_recording().drop(columns="Terrain")
    csv_path = tmp_path / "recording.csv"
    frame.to_csv(csv_path, index=False)
    predictor = TerrainPredictor(ConstantPipeline(), {"source": "test"})

    direct_features, stride_rows = stride_features(frame, "left")
    result = predictor.predict_from_csv(csv_path, side="left")

    assert direct_features.shape == (2, 150)
    assert len(stride_rows) == len(result) == 2
    assert result["predicted_terrain"].tolist() == [1, 1]
    assert result["probability_1"].tolist() == [0.8, 0.8]
    assert result["step"].tolist() == [1, 2]
    right_features, _ = stride_features(frame, "right")
    assert not np.array_equal(right_features, direct_features)

    with pytest.raises(ValueError, match="side must"):
        stride_features(frame, "unknown")
    with pytest.raises(ValueError, match="Terrain is needed"):
        stride_features(frame, "left", require_terrain=True)
    with pytest.raises(ValueError, match="150 columns"):
        predictor.predict(np.ones((1, 149)))


def test_saved_model_and_group_audit() -> None:
    if not (REPORT / "terrain_predictor.joblib").is_file():
        pytest.skip("Local validation artifact is absent")

    import joblib  # type: ignore[import-untyped]

    metadata = json.loads(
        (REPORT / "terrain_predictor_metadata.json").read_text(encoding="utf-8")
    )
    config = json.loads((REPORT / "experiment_config.json").read_text(encoding="utf-8"))
    predictor = joblib.load(REPORT / "terrain_predictor.joblib")
    assert isinstance(predictor, TerrainPredictor)
    assert predictor.feature_names == tuple(metadata["feature_names"])
    assert predictor.classes_ == tuple(metadata["terrain_codes"])
    assert len(predictor.feature_names) == 140
    assert not set(metadata["training_people"]) & set(metadata["heldout_people"])
    assert config["heldout_group"] == metadata["heldout_people"]
    assert cast(Any, predictor.pipeline).named_steps["scaler"].n_features_in_ == 140

    raw_csv = (
        ROOT.parent
        / "research-publication-data"
        / "data"
        / "raw"
        / "ntnu"
        / "P1"
        / "FLwi01PS.csv"
    )
    if raw_csv.is_file():
        frame = pd.read_csv(raw_csv).drop(columns=["Terrain", "Turn"], errors="ignore")
        features, _ = stride_features(frame, "left")
        result = predictor.predict_from_csv(frame, side="left")
        expected = predictor.predict(features[:, predictor.selected_indices])
        np.testing.assert_array_equal(result["predicted_terrain"], expected)
        probability_columns = [f"probability_{code}" for code in predictor.classes_]
        np.testing.assert_allclose(
            result[probability_columns].sum(axis=1), 1.0, rtol=1e-6
        )

    assignments: dict[str, set[str]] = {}
    groups: dict[str, set[str]] = {}
    with (REPORT / "fold_assignments.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            assignments.setdefault(row["participant"], set()).add(row["fold"])
            groups.setdefault(row["participant"], set()).add(row["recording_group"])
    assert len(assignments) == 20
    assert all(len(folds) == 1 for folds in assignments.values())
    assert groups["P7"] == {"P7_TF", "P7_TT"}
    assert assignments["P7"] == {"2"}

    with (REPORT / "holdout_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        holdout = list(csv.DictReader(stream))
    accuracy = sum(row["true"] == row["predicted"] for row in holdout) / len(holdout)
    assert accuracy == pytest.approx(config["holdout_metrics"]["accuracy"])
