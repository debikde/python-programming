"""Audit the saved participant-level evaluation, when local data is available."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT.parent
    / "research-publication-data"
    / "artifacts"
    / "reports"
    / "model_experiments"
)
BASELINE = ROOT / "artifacts" / "reports" / "reproduction_wassall_2026"


def read_rows(name: str) -> list[dict[str, str]]:
    with (REPORT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module", autouse=True)
def require_local_artifacts() -> None:
    if not (REPORT / "experiment_config.json").is_file():
        pytest.skip("Local research data and generated experiment artifacts are absent")


def test_participant_folds_keep_both_p7_recordings_together() -> None:
    config = json.loads((REPORT / "experiment_config.json").read_text(encoding="utf-8"))
    assignments = read_rows("fold_assignments.csv")
    by_person: dict[str, set[str]] = defaultdict(set)
    p7_groups: set[str] = set()
    for row in assignments:
        by_person[row["participant"]].add(row["fold"])
        if row["participant"] == "P7":
            p7_groups.add(row["recording_group"])
    assert len(by_person) == config["people"] == 20
    assert all(len(folds) == 1 for folds in by_person.values())
    assert p7_groups == {"P7_TT", "P7_TF"}
    assert by_person["P7"] == {"2"}
    for fold, people in enumerate(config["physical_lso_groups"], start=1):
        assigned = {
            person for person, values in by_person.items() if str(fold) in values
        }
        assert assigned == set(people)


def test_baseline_reproduces_previous_folds() -> None:
    baseline_folds = BASELINE / "tables" / "fold_results.csv"
    with baseline_folds.open(encoding="utf-8-sig", newline="") as stream:
        old = [row for row in csv.DictReader(stream) if row["experiment"] == "8_LSO"]
    new = [
        row
        for row in read_rows("fold_results.csv")
        if row["experiment"] == "E1-A: baseline"
    ]
    assert len(old) == len(new) == 5
    for before, after in zip(
        sorted(old, key=lambda row: int(row["fold"])),
        sorted(new, key=lambda row: int(row["fold"])),
        strict=True,
    ):
        assert abs(float(before["accuracy"]) - float(after["accuracy"])) < 1e-10
        assert int(before["test_cycles"]) == int(after["test_cycles"])


def test_test_cycles_are_paired_and_fold_accuracy_matches_predictions() -> None:
    predictions = read_rows("predictions.csv")
    fold_rows = read_rows("fold_results.csv")
    by_experiment: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    by_fold: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        observation = tuple(
            row[key] for key in ("fold", "recording_group", "trial", "step", "run")
        )
        by_experiment[row["experiment"]].add(observation)
        by_fold[(row["experiment"], row["fold"])].append(row)
    baseline = by_experiment["E1-A: baseline"]
    for name in (
        "E1-B: stride QC",
        "E1-C: без P16/P19 в train",
        "E2-B: 50 точек",
        "E2-C: статистики",
        "E3-A: Acc+Gyr",
        "E3-B: +FreeAcc",
        "E3-D: PS+TH",
    ):
        assert by_experiment[name] == baseline
    assert by_experiment["E1-D: без кодов 8/9"] < baseline
    for fold in fold_rows:
        rows = by_fold[(fold["experiment"], fold["fold"])]
        assert len(rows) == int(fold["test_cycles"])
        correct = sum(row["true"] == row["predicted"] for row in rows)
        assert abs(correct / len(rows) - float(fold["accuracy"])) < 1e-12


def test_config_excludes_labels_and_identifiers_from_features() -> None:
    config = json.loads((REPORT / "experiment_config.json").read_text(encoding="utf-8"))
    assert config["cycles"] == config["ps_th_paired_cycles"] == 9781
    assert not {"Terrain", "Steps", "participant", "recording_group", "trial"} & set(
        config["channels_full"]
    )
    experiments = read_rows("experiment_results.csv")
    counts = {row["Конфигурация"]: int(row["Признаков"]) for row in experiments}
    assert counts["E1-A: baseline"] == 300
    assert counts["E2-B: 50 точек"] == 1500
    assert counts["E2-C: статистики"] == 210
    assert counts["E3-A: Acc+Gyr"] == 120
    assert counts["E3-D: PS+TH"] == 600
