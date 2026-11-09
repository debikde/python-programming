"""Обучить рабочую модель на 20 людях по параметрам из ноутбука 04.

Независимая оценка относится к отдельной модели, обученной на 16 людях.
"""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from research_graph.api.inference import DEFAULT_ARTIFACT_DIR
from research_graph.models.terrain_predictor import (
    TERRAIN_CODES,
    TerrainPredictor,
    stride_features,
)
from research_graph.paths import NTNU_DIR, PROJECT_ROOT
from research_graph.pipelines.preprocessing import RIGHT_GROUPS

DATA_DIR = NTNU_DIR
ROOT = PROJECT_ROOT


def main() -> None:
    report = DEFAULT_ARTIFACT_DIR
    config = json.loads((report / "experiment_config.json").read_text(encoding="utf-8"))
    saved_metadata = json.loads(
        (report / "terrain_predictor_metadata.json").read_text(encoding="utf-8")
    )
    experimental = joblib.load(report / "terrain_predictor.joblib")
    if not isinstance(experimental, TerrainPredictor):
        raise TypeError("The experimental artifact has an unexpected type")
    if experimental.feature_names != tuple(config["selected_feature_names"]):
        raise ValueError("Feature schema differs from notebook 04")
    if (
        hashlib.sha256((report / "terrain_predictor.joblib").read_bytes()).hexdigest()
        != saved_metadata["artifact_sha256"]
    ):
        raise ValueError("The experimental artifact checksum changed")

    comparison = pd.read_csv(report / "experiment_results.csv")
    algorithms = comparison[
        comparison["Конфигурация"].isin(
            [
                "N2: train-only StandardScaler",
                "M2: Logistic Regression",
                "M3: Extra Trees",
            ]
        )
    ]
    participant = pd.read_csv(report / "holdout_predictions.csv")
    by_person = (
        participant.assign(correct=participant["true"].eq(participant["predicted"]))
        .groupby("participant", sort=True)
        .agg(cycles=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    recall = pd.read_csv(report / "class_recall.csv")
    holdout_recall = recall[
        recall["experiment"].eq("Отложенная группа: выбранная модель")
    ]
    summary = {
        "source_notebook": "notebooks/04_model_validation.ipynb",
        "dataset_sha256": config["dataset_sha256"],
        "unit": "step_cycle",
        "classes": list(TERRAIN_CODES),
        "selection": {
            "algorithm": config["selected_algorithm"],
            "parameter_max_features": config["selected_parameter"],
            "feature_set": config["selected_feature_set"],
            "feature_count": len(experimental.feature_names),
            "based_only_on": "four development-group folds (16 people)",
        },
        "experimental_model": {
            "filename": "terrain_predictor.joblib",
            "sha256": saved_metadata["artifact_sha256"],
            "training_people": saved_metadata["training_people"],
            "heldout_people": saved_metadata["heldout_people"],
            "holdout_metrics": config["holdout_metrics"],
            "by_person": by_person,
            "holdout_class_recall": {
                str(int(row["class"])): float(row["recall"])
                for _, row in holdout_recall.iterrows()
            },
        },
        "five_fold_lso_fixed_configs": [
            {
                "algorithm": row["Конфигурация"],
                "accuracy": float(row["Accuracy"]),
                "macro_f1": float(row["Macro-F1"]),
                "balanced_accuracy": float(row["Balanced accuracy"]),
                "roc_auc_ovr": float(row["ROC-AUC OvR"]),
            }
            for _, row in algorithms.iterrows()
        ],
        "limits": [
            "The four held-out people appeared in earlier project notebooks.",
            "Codes 6 and 7 have zero recall on this held-out group.",
            "Terrain 8/9 names and some NTNU labels remain uncertain.",
        ],
    }
    (ROOT / "docs" / "lab10_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = json.loads(
        (DATA_DIR / "import_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["dataset_sha256"] != config["dataset_sha256"]:
        raise ValueError("Dataset differs from the evaluated dataset")
    hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    vectors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    people: set[str] = set()
    seen: dict[str, set[str]] = {}
    for path in sorted(DATA_DIR.glob("P*/*PS.csv")):
        group = path.parent.name
        relative = path.relative_to(DATA_DIR).as_posix()
        digest = hashes[relative]
        if digest in seen.setdefault(group, set()):
            continue
        seen[group].add(digest)
        frame = pd.read_csv(path)
        side = "right" if group in RIGHT_GROUPS else "left"
        features, strides = stride_features(frame, side, require_terrain=True)
        vectors.append(features[:, experimental.selected_indices])
        labels.append(strides["terrain"].to_numpy(dtype=int))
        people.add(group.split("_")[0])
    x = np.vstack(vectors)
    y = np.concatenate(labels)
    if x.shape != (9781, 140) or len(people) != 20:
        raise ValueError("Cycle count or participant count differs from notebook 04")
    pipeline = clone(experimental.pipeline)
    pipeline.fit(x, y)
    metadata = dict(experimental.metadata)
    metadata.update(
        {
            "purpose": "working_model_trained_on_all_20_people",
            "training_people": sorted(people, key=lambda value: int(value[1:])),
            "heldout_people": [],
            "validation": {
                "reference": "experimental_16_people",
                "same_model_fit": False,
                "independent_test_for_this_artifact": False,
            },
        }
    )
    working = TerrainPredictor(pipeline, metadata, experimental.selected_indices)
    target = report / "terrain_predictor_final.joblib"
    joblib.dump(working, target, compress=3)
    final_metadata = {
        "purpose": metadata["purpose"],
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "filename": target.name,
        "training_people": metadata["training_people"],
        "feature_count": 140,
        "independent_test_for_this_artifact": False,
    }
    (report / "terrain_predictor_final_metadata.json").write_text(
        json.dumps(final_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Validation frozen in docs/lab10_validation.json")
    print("Working model:", target)
    print("SHA-256:", final_metadata["sha256"])


if __name__ == "__main__":
    main()
