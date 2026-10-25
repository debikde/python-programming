"""Trusted artifact loading and cycle-level prediction helpers."""

import hashlib
import io
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from research_graph.models.terrain_predictor import (
    FEATURE_NAMES,
    SOURCE_CHANNELS,
    TERRAIN_CODES,
    TerrainPredictor,
    stride_features,
)

MAX_MODEL_BYTES = 20 * 1024 * 1024
MAX_CSV_BYTES = 10 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT.parent
    / "research-publication-data"
    / "artifacts"
    / "reports"
    / "model_validation"
)


@dataclass(frozen=True)
class TrustedArtifact:
    name: str
    sha256: str
    filename: str


def read_registry(path: Path) -> dict[str, TrustedArtifact]:
    """Registry is repository-controlled; uploads cannot add trusted hashes."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    registry = {item["name"]: TrustedArtifact(**item) for item in raw["models"]}
    if any(Path(item.filename).name != item.filename for item in registry.values()):
        raise ValueError("Registered model filenames must not contain a path")
    return registry


def validate_model(model: object) -> TerrainPredictor:
    """Reject old or incompatible fitted objects before activation."""
    if not isinstance(model, TerrainPredictor):
        raise ValueError("Artifact is not a TerrainPredictor")
    expected = tuple(i for i, name in enumerate(FEATURE_NAMES) if "cadence" not in name)
    if model.selected_indices != expected or len(model.feature_names) != 140:
        raise ValueError("Artifact does not use the 140-feature schema")
    if model.feature_names != tuple(FEATURE_NAMES[i] for i in expected):
        raise ValueError("Feature names do not match the saved schema")
    if model.classes_ != TERRAIN_CODES:
        raise ValueError("Terrain class order is incompatible")
    pipeline = model.pipeline
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Artifact has no fitted sklearn Pipeline")
    steps = cast(Any, pipeline).named_steps
    if not isinstance(steps.get("scaler"), StandardScaler):
        raise ValueError("Artifact has no train-only StandardScaler")
    if not isinstance(steps.get("classifier"), RandomForestClassifier):
        raise ValueError("Artifact is not the selected Random Forest")
    if steps["scaler"].n_features_in_ != 140:
        raise ValueError("Scaler was fitted on a different feature schema")
    metadata = model.metadata
    if metadata.get("feature_names") != list(model.feature_names):
        raise ValueError("Artifact metadata and feature schema differ")
    if metadata.get("normalization") != "StandardScaler_fit_on_training_strides_only":
        raise ValueError("Artifact normalization metadata is incompatible")
    versions = metadata.get("versions")
    if not isinstance(versions, dict) or versions.get("scikit-learn") != version(
        "scikit-learn"
    ):
        raise ValueError("Artifact was fitted with a different scikit-learn version")
    return model


def activate_trusted_model(
    uploaded: bytes, registry: dict[str, TrustedArtifact], artifact_dir: Path
) -> tuple[str, TerrainPredictor]:
    """Load only a locally registered file after both hashes match the upload."""
    if not uploaded or len(uploaded) > MAX_MODEL_BYTES:
        raise ValueError("Model file is empty or exceeds 20 MB")
    digest = hashlib.sha256(uploaded).hexdigest()
    matched = next((item for item in registry.values() if item.sha256 == digest), None)
    if matched is None:
        raise ValueError("Unknown model checksum; arbitrary joblib files are rejected")
    local_path = artifact_dir / matched.filename
    if not local_path.is_file() or local_path.stat().st_size > MAX_MODEL_BYTES:
        raise ValueError("Registered local model is missing or too large")
    if hashlib.sha256(local_path.read_bytes()).hexdigest() != digest:
        raise ValueError("Registered local model checksum differs from the upload")
    model = validate_model(joblib.load(local_path))
    return matched.name, model


def classify_matrix(
    model: TerrainPredictor, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Use the fitted scaler and classifier in their original order."""
    predicted = model.predict(matrix)
    probabilities = model.pipeline.predict_proba(matrix)
    return predicted, probabilities


def metric_summary(
    truth: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | None]:
    """Match the notebook's eight-class macro metrics where defined."""
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth, predicted, labels=TERRAIN_CODES, average="macro", zero_division=0
            )
        ),
        "balanced_accuracy": (
            float(balanced_accuracy_score(truth, predicted))
            if len(np.unique(truth)) > 1
            else None
        ),
        "roc_auc_ovr": None,
    }
    if set(np.unique(truth)) == set(TERRAIN_CODES):
        result["roc_auc_ovr"] = float(
            roc_auc_score(
                truth,
                probabilities,
                labels=TERRAIN_CODES,
                multi_class="ovr",
                average="macro",
            )
        )
    return result


def parse_csv(payload: bytes) -> pd.DataFrame:
    if not payload or len(payload) > MAX_CSV_BYTES:
        raise ValueError("CSV is empty or exceeds 10 MB")
    try:
        frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    except (pd.errors.ParserError, UnicodeError, ValueError) as error:
        raise ValueError("CSV must be a valid UTF-8 table") from error
    if frame.empty:
        raise ValueError("CSV has no data rows")
    return frame


def predict_csv(
    model: TerrainPredictor, frame: pd.DataFrame, *, side: str | None
) -> dict[str, object]:
    """Accept PS samples or a table of 140/150 stride-level features."""
    raw_columns = set(SOURCE_CHANNELS) | {"Steps"}
    if raw_columns.issubset(frame.columns):
        if side not in {"left", "right"}:
            raise ValueError("Raw IMU CSV requires side='left' or side='right'")
        features, strides = stride_features(frame, side)
        if len(features) == 0:
            raise ValueError("No valid step cycles in CSV")
        matrix = features[:, model.selected_indices]
        details = strides.to_dict(orient="records")
        labels: list[int | None] = [None] * len(details)
        if "Terrain" in frame:
            _, labelled = stride_features(frame, side, require_terrain=True)
            by_run = (
                dict(zip(labelled["run"], labelled["terrain"], strict=True))
                if len(labelled)
                else {}
            )
            labels = [
                int(by_run[row["run"]]) if row["run"] in by_run else None
                for row in details
            ]
        kind = "imu_cycles"
    else:
        if not set(model.feature_names).issubset(frame.columns):
            raise ValueError(
                "CSV needs PS channels and Steps, or the 140 named features"
            )
        if len(frame) > 10000:
            raise ValueError("Prepared CSV exceeds 10000 cycles")
        matrix = frame.loc[:, list(model.feature_names)].to_numpy(dtype="float32")
        details = [{"row": int(i)} for i in range(len(frame))]
        if "Terrain" in frame:
            values = pd.to_numeric(frame["Terrain"], errors="coerce")
            labels = [int(v) if v in TERRAIN_CODES else None for v in values]
        else:
            labels = [None] * len(frame)
        kind = "prepared_cycles"
    predicted, probabilities = classify_matrix(model, matrix)
    rows = []
    for index, detail in enumerate(details):
        rows.append(
            {
                **detail,
                "true_terrain": labels[index],
                "predicted_terrain": int(predicted[index]),
                "probabilities": {
                    str(code): float(probabilities[index, class_index])
                    for class_index, code in enumerate(model.classes_)
                },
            }
        )
    valid = np.array([index for index, value in enumerate(labels) if value is not None])
    metrics = None
    if len(valid):
        truth = np.array([labels[index] for index in valid], dtype=int)
        metrics = metric_summary(truth, predicted[valid], probabilities[valid])
    return {
        "unit": "step_cycle",
        "input_kind": kind,
        "rows": rows,
        "metrics": metrics,
        "labelled_cycles": len(valid),
    }
