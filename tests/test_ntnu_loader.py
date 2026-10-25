"""Archive import integrity and filesystem safety checks using synthetic files."""

import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from research_graph.sources.ntnu_loader import (
    import_dataset,
    participant_id,
    recording_trial_id,
)


def make_archive(path: Path, member: str = "FLwi01PS.csv") -> None:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as inner:
        inner.writestr(member, "value\n1\n")
    with ZipFile(path, "w") as outer:
        outer.writestr("P1.zip", buffer.getvalue())
        outer.writestr("0_Readme.txt", "Synthetic test fixture")
        outer.writestr("MANIFEST.TXT", "Synthetic source manifest")


def test_import_and_rerun_preserve_bytes(tmp_path: Path) -> None:
    archive, destination = tmp_path / "input.zip", tmp_path / "raw"
    make_archive(archive)
    first = import_dataset(archive, destination)
    assert first == import_dataset(archive, destination)
    assert (destination / "P1/FLwi01PS.csv").read_bytes() == b"value\n1\n"
    manifest = json.loads((destination / "import_manifest.json").read_text())
    assert manifest["participants"] == ["P1"]
    assert len(manifest["source_archive_sha256"]) == 64


@pytest.mark.parametrize(
    "member", ["../outside.csv", "/outside.csv", "C:/outside.csv", "..\\outside.csv"]
)
def test_rejects_path_traversal(tmp_path: Path, member: str) -> None:
    archive = tmp_path / "input.zip"
    make_archive(archive, member)
    with pytest.raises(ValueError, match="Unsafe"):
        import_dataset(archive, tmp_path / "raw")
    assert not (tmp_path / "outside.csv").exists()


def test_does_not_overwrite_modified_source(tmp_path: Path) -> None:
    archive, destination = tmp_path / "input.zip", tmp_path / "raw"
    make_archive(archive)
    import_dataset(archive, destination)
    target = destination / "P1/FLwi01PS.csv"
    target.write_text("keep this", encoding="utf-8")
    with pytest.raises(FileExistsError):
        import_dataset(archive, destination)
    assert target.read_text(encoding="utf-8") == "keep this"


def test_rejects_unexpected_member(tmp_path: Path) -> None:
    archive = tmp_path / "input.zip"
    make_archive(archive, "program.exe")
    with pytest.raises(ValueError, match="Unexpected"):
        import_dataset(archive, tmp_path / "raw")


def test_merge_preserves_both_downloads_and_manifests(tmp_path: Path) -> None:
    destination = tmp_path / "raw"
    for number in (1, 2):
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as inner:
            inner.writestr("FLwi01PS.csv", f"value\n{number}\n")
        archive = tmp_path / f"part{number}.zip"
        with ZipFile(archive, "w") as outer:
            outer.writestr(f"P{number}.zip", buffer.getvalue())
            outer.writestr("MANIFEST.TXT", f"Part {number}")
        import_dataset(archive, destination)
    manifest_path = destination / "import_manifest.json"
    first_manifest = manifest_path.read_bytes()
    import_dataset(tmp_path / "part2.zip", destination)
    assert manifest_path.read_bytes() == first_manifest
    manifest = json.loads(first_manifest)
    assert manifest["participants"] == ["P1", "P2"]
    assert len(manifest["source_archives"]) == 2
    assert len(list(destination.glob("_imports/*/MANIFEST.TXT"))) == 2
    assert (destination / "P1/FLwi01PS.csv").read_bytes() == b"value\n1\n"
    assert (destination / "P2/FLwi01PS.csv").read_bytes() == b"value\n2\n"


def test_p7_leg_views_share_participant_but_not_trial_key() -> None:
    assert participant_id("P7_TT") == participant_id("P7_TF") == "P7"
    assert recording_trial_id("P7_TT", "FLwi01") == "TT_FLwi01"
    assert recording_trial_id("P7_TF", "FLwi01") == "TF_FLwi01"
    assert recording_trial_id("P1", "FLwi01") == "FLwi01"


def test_rejects_unknown_recording_group() -> None:
    with pytest.raises(ValueError, match="Unexpected recording group"):
        participant_id("unknown")


def test_legacy_manifest_keeps_original_provenance(tmp_path: Path) -> None:
    archive, destination = tmp_path / "input.zip", tmp_path / "raw"
    make_archive(archive)
    import_dataset(archive, destination)
    path = destination / "import_manifest.json"
    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy.pop("source_archives")
    legacy.pop("dataset_sha256")
    for record in legacy["files"]:
        if record["path"].endswith("MANIFEST.TXT"):
            (destination / record["path"]).replace(destination / "MANIFEST.TXT")
            record["path"] = "MANIFEST.TXT"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    import_dataset(archive, destination)
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert len(migrated["source_archives"]) == 1
    assert migrated["source_archive_sha256"] == legacy["source_archive_sha256"]
    assert len(migrated["dataset_sha256"]) == 64
    assert [record["path"] for record in migrated["files"]].count("MANIFEST.TXT") == 1
    assert not list(destination.glob("_imports/*/MANIFEST.TXT"))
