"""Создаёт небольшие искусственные примеры для ручной проверки ЛР10."""

import csv
import json
import math
from pathlib import Path

from research_graph.models.terrain_predictor import FEATURE_NAMES, TERRAIN_CODES
from research_graph.pipelines.preprocessing import SOURCE_CHANNELS


def write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def main() -> None:
    folder = Path(__file__).resolve().parents[1] / "examples" / "lab10"
    folder.mkdir(parents=True, exist_ok=True)

    feature_names = [name for name in FEATURE_NAMES if "cadence" not in name]
    prepared = []
    for cycle, terrain in enumerate(TERRAIN_CODES):
        values = [round(0.1 * math.sin(cycle + index / 10), 6) for index in range(140)]
        prepared.append([*values, terrain, f"пример, {cycle + 1}"])
    write_csv(
        folder / "prepared_labeled.csv", [*feature_names, "Terrain", "note"], prepared
    )
    write_csv(
        folder / "prepared_unlabeled.csv",
        feature_names,
        [row[:140] for row in prepared[:2]],
    )
    (folder / "records.json").write_text(
        json.dumps({"records": [row[:140] for row in prepared[:2]]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (folder / "manual_records.json").write_text(
        json.dumps([row[:140] for row in prepared[:2]], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (folder / "invalid_records.json").write_text(
        '{"records": [[1, 2]]}\n', encoding="utf-8"
    )

    imu_rows = []
    for step, terrain in enumerate(TERRAIN_CODES, start=1):
        for sample in range(35):
            values = [
                round(math.sin(2 * math.pi * sample / 35 + index / 7) + step / 20, 6)
                for index in range(len(SOURCE_CHANNELS))
            ]
            imu_rows.append([*values, step, terrain])
    write_csv(
        folder / "imu_labeled.csv", [*SOURCE_CHANNELS, "Steps", "Terrain"], imu_rows
    )
    write_csv(folder / "invalid_columns.csv", ["wrong_column"], [[1]])
    (folder / "untrusted_model.joblib").write_bytes(b"Not a trusted model artifact.\n")


if __name__ == "__main__":
    main()
