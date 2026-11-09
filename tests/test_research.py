"""Checks for the data import and signal steps used by the notebooks."""

import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from typer.testing import CliRunner

from research_graph.cli import app
from research_graph.config import Settings, get_settings
from research_graph.models.terrain_predictor import SOURCE_CHANNELS, stride_features
from research_graph.pipelines.features import raw_and_person_channels, stride_statistics
from research_graph.pipelines.preprocessing import (
    MIRROR_CHANNELS,
    cadence_from_steps,
    filter_imu,
    person_mean_and_std,
)
from research_graph.sources.ntnu_loader import (
    import_dataset,
    participant_id,
    recording_trial_id,
)


def make_archive(path: Path, group: str, name: str, content: bytes) -> None:
    """Build a small archive with the same two ZIP levels as NTNU."""
    inner_data = io.BytesIO()
    with ZipFile(inner_data, "w") as inner:
        inner.writestr(name, content)
    with ZipFile(path, "w") as outer:
        outer.writestr("0_Readme.txt", b"NTNU test archive")
        outer.writestr(f"{group}.zip", inner_data.getvalue())


def sample_recording() -> pd.DataFrame:
    samples = np.arange(60, dtype=float)
    frame = pd.DataFrame(
        {name: samples / 100 + index + 1 for index, name in enumerate(SOURCE_CHANNELS)}
    )
    frame["Steps"] = np.repeat([1, 2, 3], 20)
    frame["Terrain"] = np.repeat([1, 2, 99], 20)
    return frame


def test_import_keeps_csv_bytes_and_combines_participants(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    destination = tmp_path / "data"
    csv_bytes = b"Steps,Terrain\r\n1,1\r\n"
    make_archive(first, "P7_TT", "walkPS.csv", csv_bytes)
    make_archive(second, "P7_TF", "walkPS.csv", b"Steps,Terrain\n2,2\n")

    imported = import_dataset(first, destination)
    assert any(item.path == "P7_TT/walkPS.csv" for item in imported)
    assert (destination / "P7_TT" / "walkPS.csv").read_bytes() == csv_bytes
    assert import_dataset(first, destination) == imported
    import_dataset(second, destination)

    manifest = json.loads((destination / "import_manifest.json").read_text())
    assert manifest["participants"] == ["P7_TF", "P7_TT"]
    assert len(manifest["source_archives"]) == 2
    assert {item["path"] for item in manifest["files"]} >= {
        "P7_TT/walkPS.csv",
        "P7_TF/walkPS.csv",
    }
    assert (
        manifest["source_archive_sha256"]
        == hashlib.sha256(first.read_bytes()).hexdigest()
    )

    (destination / "P7_TT" / "walkPS.csv").write_bytes(b"locally changed")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        import_dataset(first, destination)


@pytest.mark.parametrize(
    ("member", "message"),
    [
        ("../escape.csv", "Unsafe archive member"),
        ("other/track.csv", "Unsafe archive member"),
        ("P1.txt", "Unexpected top-level member"),
    ],
)
def test_import_rejects_unexpected_outer_members(
    tmp_path: Path, member: str, message: str
) -> None:
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as outer:
        outer.writestr(member, b"bad")
    with pytest.raises(ValueError, match=message):
        import_dataset(archive, tmp_path / "data")
    assert not (tmp_path / "escape.csv").exists()


def test_import_rejects_duplicate_inner_names(tmp_path: Path) -> None:
    inner_data = io.BytesIO()
    with ZipFile(inner_data, "w") as inner:
        inner.writestr("Walk.csv", b"first")
        inner.writestr("walk.CSV", b"second")
    archive = tmp_path / "duplicate.zip"
    with ZipFile(archive, "w") as outer:
        outer.writestr("P1.zip", inner_data.getvalue())
    with pytest.raises(ValueError, match="Duplicate"):
        import_dataset(archive, tmp_path / "data")


def test_p7_views_belong_to_one_person() -> None:
    assert participant_id("P7_TT") == participant_id("P7_TF") == "P7"
    assert recording_trial_id("P7_TF", "FLwi01") == "TF_FLwi01"
    assert recording_trial_id("P3", "FLwi01") == "FLwi01"
    with pytest.raises(ValueError, match="Unexpected recording group"):
        participant_id("P7_unknown")


def test_signal_preparation_and_cycle_features() -> None:
    frame = sample_recording()
    original, filtered = filter_imu(frame, right=False)
    mirrored, _ = filter_imu(frame, right=True)
    assert original.shape == filtered.shape == (60, 17)
    for name in MIRROR_CHANNELS:
        index = SOURCE_CHANNELS.index(name)
        np.testing.assert_allclose(mirrored[:, index], -original[:, index])
    np.testing.assert_allclose(cadence_from_steps(frame["Steps"].to_numpy()), 5)

    features, cycles = stride_features(frame, "left", require_terrain=True)
    assert features.shape == (2, 150)
    assert cycles["terrain"].tolist() == [1, 2]
    assert cycles["samples"].tolist() == [20, 20]
    assert len(stride_features(frame, "right")[1]) == 3

    with pytest.raises(ValueError, match="side"):
        stride_features(frame, "unknown")
    with pytest.raises(ValueError, match="Terrain"):
        stride_features(frame.drop(columns="Terrain"), "left", require_terrain=True)
    with pytest.raises(ValueError, match="Missing CSV columns"):
        stride_features(frame.drop(columns="Acc_X"), "left")
    with pytest.raises(ValueError, match="Non-finite"):
        filter_imu(frame.assign(Acc_X=np.nan), right=False)


def test_normalization_and_summary_features() -> None:
    samples = np.tile(np.arange(1, 18, dtype=float), (4, 1))
    mean, standard_deviation = person_mean_and_std(samples)
    np.testing.assert_array_equal(mean, samples[0])
    np.testing.assert_array_equal(standard_deviation, np.ones(17))
    combined = raw_and_person_channels(samples, np.ones(4), np.ones(18), np.ones(18))
    assert combined.shape == (4, 30)
    assert stride_statistics(np.arange(60, dtype=float).reshape(4, 15)).shape == (105,)


def test_cli_settings_and_safe_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in os.environ:
        if name.startswith("RG_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    runner = CliRunner()
    try:
        assert Settings().max_concurrency == 4
        assert runner.invoke(app, ["version"]).stdout.strip() == "0.1.0"
        result = runner.invoke(app, ["check-config"])
        assert result.exit_code == 0
        assert "github_token_present=false" in result.stdout
        assert (tmp_path / "data").is_dir()

        secret = "test_secret_value"
        monkeypatch.setenv("RG_GITHUB_TOKEN", secret)
        monkeypatch.setenv("RG_DATA_DIR", secret)
        get_settings.cache_clear()
        result = runner.invoke(app, ["check-config"])
        assert result.exit_code == 0
        assert secret not in result.output
        assert "[REDACTED]" in result.stdout

        monkeypatch.setenv("RG_MAX_CONCURRENCY", "0")
        get_settings.cache_clear()
        result = runner.invoke(app, ["check-config"])
        assert result.exit_code == 2
        assert secret not in result.output
    finally:
        get_settings.cache_clear()
