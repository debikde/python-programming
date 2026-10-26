"""Признаки цикла шага и сохранённая модель Terrain."""

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from research_graph.pipelines.preprocessing import SOURCE_CHANNELS as SOURCE_CHANNELS
from research_graph.pipelines.preprocessing import (
    USED_CHANNELS,
    USED_INDICES,
    cadence_from_steps,
    filter_imu,
)

TERRAIN_CODES = (1, 2, 4, 5, 6, 7, 8, 9)
FEATURE_NAMES = tuple(
    f"t{point + 1:02d}_{name}"
    for point in range(10)
    for name in ("cadence_rate_hz", *USED_CHANNELS)
)


def prepare_sample_channels(
    frame: pd.DataFrame, side: str
) -> tuple[np.ndarray, np.ndarray]:
    """Filter one PS recording and return 15 raw channels plus step IDs."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    missing = set(SOURCE_CHANNELS) | {"Steps"}
    missing -= set(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    signals = frame.loc[:, list(SOURCE_CHANNELS)].to_numpy(dtype=float, copy=True)
    steps = pd.to_numeric(frame["Steps"], errors="raise").to_numpy()
    if (
        len(frame) < 10
        or not np.isfinite(signals).all()
        or not np.isfinite(steps).all()
    ):
        raise ValueError("Recording is too short or contains non-finite values")
    _, filtered = filter_imu(frame, right=side == "right")
    cadence = cadence_from_steps(steps)
    selected = np.column_stack((cadence, filtered[:, USED_INDICES]))
    return selected, steps


def stride_features(
    frame: pd.DataFrame, side: str, *, require_terrain: bool = False
) -> tuple[np.ndarray, pd.DataFrame]:
    """Convert a PS CSV into 150 features for each valid contiguous stride."""
    selected, steps = prepare_sample_channels(frame, side)
    if require_terrain and "Terrain" not in frame:
        raise ValueError("Terrain is needed when preparing labelled training data")
    terrain = frame["Terrain"].to_numpy() if require_terrain else None
    bounds = np.flatnonzero(np.r_[True, steps[1:] != steps[:-1], True])
    vectors: list[np.ndarray] = []
    rows: list[dict[str, int | float]] = []
    for run in range(len(bounds) - 1):
        start, stop = int(bounds[run]), int(bounds[run + 1])
        step = steps[start]
        if stop - start < 3 or step < 1:
            continue
        row: dict[str, int | float] = {
            "step": step,
            "run": run,
            "samples": stop - start,
        }
        if terrain is not None:
            values = terrain[start:stop]
            if not np.isin(values, TERRAIN_CODES).all():
                continue
            labels, counts = np.unique(values, return_counts=True)
            row["terrain"] = int(labels[np.argmax(counts)])
        sampled = resample_poly(selected[start:stop], 10, stop - start, axis=0)
        if sampled.shape != (10, 15) or not np.isfinite(sampled).all():
            raise ValueError("Invalid stride after resampling")
        vectors.append(sampled.ravel().astype("float32"))
        rows.append(row)
    matrix = np.stack(vectors) if vectors else np.empty((0, 150), dtype="float32")
    return matrix, pd.DataFrame(rows)


class TerrainPredictor:
    """Fitted sklearn pipeline plus the exact CSV-to-stride preprocessing."""

    def __init__(
        self,
        pipeline: Pipeline,
        metadata: dict[str, object],
        selected_indices: tuple[int, ...] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.metadata = metadata
        self.selected_indices = selected_indices or tuple(range(len(FEATURE_NAMES)))
        self.feature_names = tuple(
            FEATURE_NAMES[index] for index in self.selected_indices
        )
        self.classes_ = tuple(int(value) for value in pipeline.classes_)

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected a matrix with {len(self.feature_names)} columns"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Features contain NaN or infinity")
        return np.asarray(self.pipeline.predict(matrix))

    def predict_from_csv(
        self, source: str | Path | pd.DataFrame, *, side: str
    ) -> pd.DataFrame:
        frame = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
        features, strides = stride_features(frame, side)
        if len(features) == 0:
            raise ValueError("No valid strides found in the uploaded CSV")
        result = strides.copy()
        selected = features[:, self.selected_indices]
        result["predicted_terrain"] = self.predict(selected)
        probabilities = self.pipeline.predict_proba(selected)
        for index, terrain in enumerate(self.classes_):
            result[f"probability_{terrain}"] = probabilities[:, index]
        return result
