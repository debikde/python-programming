"""Import the provided NTNU archive without changing original CSV bytes."""

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

from pydantic import BaseModel, Field

MAX_EXPANDED_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class ImportedFile:
    """An extracted source file and its content checksum."""

    path: str
    size_bytes: int
    sha256: str


class ArchiveSource(BaseModel):
    """One download contributing to the combined local dataset."""

    name: str
    sha256: str
    recording_groups: list[str]
    files: list[str]


class ImportManifest(BaseModel):
    """Provenance for one or more downloads; accepts the original manifest."""

    dataset_doi: str = "10.18710/U8RGDL"
    source_archive: str
    source_archive_sha256: str
    participants: list[str]
    participant_archive_md5: dict[str, str]
    files: list[ImportedFile]
    source_archives: list[ArchiveSource] = Field(default_factory=list)
    dataset_sha256: str = ""


def participant_id(recording_group: str) -> str:
    """Keep P7_TT and P7_TF in the same physical participant group."""
    if not re.fullmatch(r"P\d+(?:_TT|_TF)?", recording_group):
        raise ValueError(f"Unexpected recording group: {recording_group}")
    return recording_group.split("_", maxsplit=1)[0]


def recording_trial_id(recording_group: str, trial: str) -> str:
    """Disambiguate the two leg views without inventing extra participants."""
    participant_id(recording_group)
    suffix = recording_group.split("_", maxsplit=1)
    return f"{suffix[1]}_{trial}" if len(suffix) == 2 else trial


def _check_member(info: ZipInfo) -> None:
    """Reject paths and special files that must not be extracted."""
    path = PurePosixPath(info.filename)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.name in {".", ".."}
        or "\\" in info.filename
        or ":" in info.filename
        or ((info.external_attr >> 16) & 0o170000) == 0o120000
    ):
        raise ValueError(f"Unsafe archive member: {info.filename}")


def _save(root: Path, relative: str, content: bytes) -> ImportedFile:
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output path escapes the dataset directory")
    digest = hashlib.sha256(content).hexdigest()
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"Refusing to overwrite modified source: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return ImportedFile(relative, len(content), digest)


def import_dataset(archive: Path, destination: Path) -> list[ImportedFile]:
    """Extract nested participant ZIPs, verify CRC, and write a provenance manifest.

    Reruns keep identical files and reject modified originals. This imports the
    supplied subset only; no network access or data cleaning takes place.
    """
    records: list[ImportedFile] = []
    participants: list[str] = []
    archive_checksums: dict[str, str] = {}
    expanded = 0
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest_path = destination / "import_manifest.json"
    previous = (
        ImportManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    with ZipFile(archive) as outer:
        if sum(info.file_size for info in outer.infolist()) > MAX_EXPANDED_BYTES:
            raise ValueError("Archive exceeds the expanded-size limit")
        for info in sorted(outer.infolist(), key=lambda item: item.filename):
            _check_member(info)
            if info.filename in {"0_Readme.txt", "MANIFEST.TXT"}:
                relative = info.filename
                if info.filename == "MANIFEST.TXT":
                    relative = f"_imports/{archive_sha256}/MANIFEST.TXT"
                    if (
                        previous
                        and previous.source_archive_sha256 == archive_sha256
                        and any(
                            record.path == "MANIFEST.TXT" for record in previous.files
                        )
                    ):
                        relative = "MANIFEST.TXT"
                records.append(_save(destination, relative, outer.read(info)))
                continue
            if not re.fullmatch(r"P\d+(?:_TT|_TF)?\.zip", info.filename):
                raise ValueError(f"Unexpected top-level member: {info.filename}")
            participant = Path(info.filename).stem
            participants.append(participant)
            content = outer.read(info)
            archive_checksums[info.filename] = hashlib.md5(
                content, usedforsecurity=False
            ).hexdigest()
            with ZipFile(io.BytesIO(content)) as inner:
                expanded += sum(item.file_size for item in inner.infolist())
                if expanded > MAX_EXPANDED_BYTES:
                    raise ValueError("Nested archives exceed the expanded-size limit")
                names: set[str] = set()
                for item in inner.infolist():
                    _check_member(item)
                    if not item.filename.lower().endswith(".csv"):
                        raise ValueError(
                            f"Unexpected participant member: {item.filename}"
                        )
                    if item.filename.casefold() in names:
                        raise ValueError("Duplicate participant archive member")
                    names.add(item.filename.casefold())
                    records.append(
                        _save(
                            destination,
                            f"{participant}/{item.filename}",
                            inner.read(item),
                        )
                    )
    old_sources = previous.source_archives if previous else []
    if previous and not old_sources:
        old_sources = [
            ArchiveSource(
                name=previous.source_archive,
                sha256=previous.source_archive_sha256,
                recording_groups=previous.participants,
                files=[record.path for record in previous.files],
            )
        ]
    sources = {source.sha256: source for source in old_sources}
    sources[archive_sha256] = ArchiveSource(
        name=archive.name,
        sha256=archive_sha256,
        recording_groups=participants,
        files=[record.path for record in records],
    )
    combined = {record.path: record for record in previous.files} if previous else {}
    combined.update({record.path: record for record in records})
    combined_files = [combined[path] for path in sorted(combined)]
    all_checksums = previous.participant_archive_md5.copy() if previous else {}
    all_checksums.update(archive_checksums)
    all_groups = sorted(
        {group for source in sources.values() for group in source.recording_groups}
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            [asdict(record) for record in combined_files],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest = ImportManifest(
        source_archive=previous.source_archive if previous else archive.name,
        source_archive_sha256=previous.source_archive_sha256
        if previous
        else archive_sha256,
        participants=all_groups,
        participant_archive_md5=all_checksums,
        files=combined_files,
        source_archives=list(sources.values()),
        dataset_sha256=fingerprint,
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "import_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return records
