"""Exercise the real FastAPI routes and the trusted-artifact boundary."""

import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from research_graph.api.inference import BUNDLED_ARTIFACT_DIR, validate_model
from research_graph.api.main import create_app
from research_graph.models.terrain_predictor import (
    FEATURE_NAMES,
    SOURCE_CHANNELS,
    TERRAIN_CODES,
    TerrainPredictor,
)


@pytest.fixture
def fitted_model(tmp_path: Path) -> tuple[TestClient, TerrainPredictor, bytes]:
    indices = tuple(i for i, name in enumerate(FEATURE_NAMES) if "cadence" not in name)
    generator = np.random.default_rng(42)
    features = generator.normal(size=(32, 140)).astype("float32")
    labels = np.tile(TERRAIN_CODES, 4)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=8, random_state=42)),
        ]
    )
    pipeline.fit(features, labels)
    model = TerrainPredictor(
        pipeline,
        {
            "feature_names": [FEATURE_NAMES[i] for i in indices],
            "normalization": "StandardScaler_fit_on_training_strides_only",
            "versions": {"scikit-learn": version("scikit-learn")},
        },
        indices,
    )
    target = tmp_path / "trusted.joblib"
    joblib.dump(model, target, compress=3)
    payload = target.read_bytes()
    registry = tmp_path / "trusted_models.json"
    registry.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "test_model",
                        "filename": target.name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return (
        TestClient(
            create_app(
                registry_path=registry, artifact_dir=tmp_path, default_model_name=None
            )
        ),
        model,
        payload,
    )


def upload(client: TestClient, payload: bytes) -> Any:
    return client.post("/upload-model", files={"file": ("model.joblib", payload)})


def sample_csv(with_terrain: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            name: np.linspace(index + 1, index + 2, 70)
            for index, name in enumerate(SOURCE_CHANNELS)
        }
    )
    frame["Steps"] = np.repeat([1, 2], 35)
    if with_terrain:
        frame["Terrain"] = np.repeat([1, 2], 35)
    return frame


def test_requires_model_and_rejects_unknown_binary(
    fitted_model: tuple[TestClient, TerrainPredictor, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = fitted_model
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model-info").json()["loaded"] is False
    assert client.post("/predict", json={"records": [[0] * 140]}).status_code == 409
    assert (
        client.post("/predict-from-csv", files={"file": ("x.csv", b"x\n1")}).status_code
        == 409
    )
    monkeypatch.setattr(
        joblib, "load", lambda _: pytest.fail("Unknown file was deserialized")
    )
    response = upload(client, b"arbitrary untrusted pickle bytes")
    assert response.status_code == 400
    assert client.get("/model-info").json()["loaded"] is False


def test_model_upload_and_prepared_prediction(
    fitted_model: tuple[TestClient, TerrainPredictor, bytes],
) -> None:
    client, model, payload = fitted_model
    response = upload(client, payload)
    assert response.status_code == 200
    assert response.json()["model_name"] == "test_model"
    assert client.get("/model-info").json()["feature_names"] == list(
        model.feature_names
    )
    row = [0.1] * 140
    result = client.post("/predict", json={"records": [row, row]})
    assert result.status_code == 200
    rows = result.json()["rows"]
    assert len(rows) == 2
    assert rows[0]["predicted_terrain"] == int(model.predict(np.array([row]))[0])
    assert sum(rows[0]["probabilities"].values()) == pytest.approx(1.0)
    assert rows[0]["input"] == dict(zip(model.feature_names, row, strict=True))
    assert client.post("/predict", json={"records": [[1, 2]]}).status_code == 422
    assert client.post("/predict", json={"records": []}).status_code == 422


def test_compatibility_rejects_wrong_library_version(
    fitted_model: tuple[TestClient, TerrainPredictor, bytes],
) -> None:
    _, model, _ = fitted_model
    model.metadata["versions"] = {"scikit-learn": "0.0"}
    with pytest.raises(ValueError, match="scikit-learn version"):
        validate_model(model)


def test_raw_csv_matches_predictor_and_handles_labels(
    fitted_model: tuple[TestClient, TerrainPredictor, bytes],
) -> None:
    client, model, payload = fitted_model
    upload(client, payload)
    frame = sample_csv()
    body = frame.to_csv(index=False).encode()
    response = client.post(
        "/predict-from-csv",
        data={"side": "left"},
        files={"file": ("walk.csv", body, "text/csv")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["unit"] == "step_cycle"
    assert result["input_kind"] == "imu_cycles"
    assert result["input_columns"] == list(model.feature_names)
    assert list(result["rows"][0]["input"]) == list(model.feature_names)
    assert len(result["rows"]) == result["labelled_cycles"] == 2
    expected = model.predict_from_csv(frame.drop(columns="Terrain"), side="left")
    assert [row["predicted_terrain"] for row in result["rows"]] == expected[
        "predicted_terrain"
    ].tolist()
    assert result["metrics"]["accuracy"] is not None
    assert result["metrics"]["roc_auc_ovr"] is None
    missing_side = client.post("/predict-from-csv", files={"file": ("walk.csv", body)})
    assert missing_side.status_code == 422

    unlabelled = sample_csv(with_terrain=False).to_csv(index=False).encode()
    response = client.post(
        "/predict-from-csv",
        data={"side": "right"},
        files={"file": ("walk.csv", unlabelled)},
    )
    assert response.status_code == 200
    assert response.json()["metrics"] is None


def test_prepared_csv_and_invalid_inputs(
    fitted_model: tuple[TestClient, TerrainPredictor, bytes],
) -> None:
    client, model, payload = fitted_model
    upload(client, payload)
    frame = pd.DataFrame(np.zeros((8, 140)), columns=model.feature_names)
    frame["Terrain"] = list(TERRAIN_CODES)
    frame["note"] = ["a,b"] + ["ok"] * 7
    body = frame.to_csv(index=False).encode()
    response = client.post("/predict-from-csv", files={"file": ("features.csv", body)})
    assert response.status_code == 200
    assert response.json()["input_kind"] == "prepared_cycles"
    assert response.json()["metrics"]["roc_auc_ovr"] is not None
    assert len(response.json()["rows"]) == 8
    assert response.json()["rows"][0]["input"]["note"] == "a,b"
    assert response.json()["input_columns"] == list(frame.columns)

    bad = client.post("/predict-from-csv", files={"file": ("bad.csv", b"x,y\n1,2")})
    assert bad.status_code == 422
    empty = client.post("/predict-from-csv", files={"file": ("empty.csv", b"")})
    assert empty.status_code == 422
    assert (
        client.post("/upload-model", files={"file": ("x", io.BytesIO(b""))}).status_code
        == 400
    )


def test_bundled_model_is_ready_and_upload_is_optional() -> None:
    assert (BUNDLED_ARTIFACT_DIR / "terrain_predictor_final.joblib").is_file()
    client = TestClient(create_app())
    info = client.get("/model-info").json()
    assert info["loaded"] is True
    assert info["model_name"] == info["default_model_name"] == "working_20_people"
    assert len(info["feature_names"]) == 140
    assert len(info["training_people"]) == 20
    response = client.post("/predict", json={"records": [[0.0] * 140]})
    assert response.status_code == 200
    assert len(response.json()["rows"]) == 1

    experimental = (BUNDLED_ARTIFACT_DIR / "terrain_predictor.joblib").read_bytes()
    assert upload(client, experimental).status_code == 200
    assert client.get("/model-info").json()["model_name"] == "working_20_people"
    selected = client.get(
        "/model-info", headers={"X-Terrain-Model": "experimental_16_people"}
    ).json()
    assert selected["loaded"] is True
    assert len(selected["training_people"]) == 16
    assert (
        client.post(
            "/predict",
            json={"records": [[0.0] * 140]},
            headers={"X-Terrain-Model": "missing"},
        ).status_code
        == 409
    )
