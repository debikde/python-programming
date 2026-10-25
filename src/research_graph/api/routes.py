"""HTTP routes for the NTNU cycle-level classifier."""

from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from research_graph.api.inference import (
    MAX_CSV_BYTES,
    MAX_MODEL_BYTES,
    activate_trusted_model,
    classify_matrix,
    parse_csv,
    predict_csv,
)
from research_graph.models.terrain_predictor import TerrainPredictor

router = APIRouter()


class PredictRequest(BaseModel):
    """One row is one prepared stride with 140 ordered numeric features."""

    records: list[list[float]] = Field(min_length=1, max_length=1000)


def current_model(request: Request) -> TerrainPredictor:
    model: TerrainPredictor | None = request.app.state.model
    if model is None:
        raise HTTPException(status_code=409, detail="Upload a trusted model first")
    return model


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/model-info")
def model_info(request: Request) -> dict[str, object]:
    model: TerrainPredictor | None = request.app.state.model
    base: dict[str, object] = {
        "loaded": model is not None,
        "trusted_models": list(request.app.state.registry),
        "unit": "step_cycle",
        "limitations": [
            "Исследовательский прототип; не предназначен для клинических решений.",
            "На четырёх отложенных участниках macro-F1 равен 0.299.",
            "Коды Terrain 8 и 9 оставлены числовыми до уточнения разметки.",
        ],
    }
    if model is not None:
        base.update(
            {
                "model_name": request.app.state.model_name,
                "feature_names": list(model.feature_names),
                "classes": list(model.classes_),
                "training_people": model.metadata.get("training_people"),
                "validation": request.app.state.validation,
            }
        )
    return base


@router.post("/upload-model")
async def upload_model(
    request: Request, file: Annotated[UploadFile, File()]
) -> dict[str, object]:
    payload = await file.read(MAX_MODEL_BYTES + 1)
    try:
        name, model = activate_trusted_model(
            payload, request.app.state.registry, request.app.state.artifact_dir
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    request.app.state.model = model
    request.app.state.model_name = name
    return {
        "status": "loaded",
        "model_name": name,
        "features": len(model.feature_names),
        "classes": list(model.classes_),
    }


@router.post("/predict")
def predict(request: Request, body: PredictRequest) -> dict[str, object]:
    model = current_model(request)
    try:
        matrix = np.asarray(body.records, dtype="float32")
        predicted, probabilities = classify_matrix(model, matrix)
    except (ValueError, OverflowError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    rows = [
        {
            "features": body.records[index],
            "predicted_terrain": int(predicted[index]),
            "probabilities": {
                str(code): float(probabilities[index, class_index])
                for class_index, code in enumerate(model.classes_)
            },
        }
        for index in range(len(matrix))
    ]
    return {"unit": "step_cycle", "rows": rows}


@router.post("/predict-from-csv")
async def predict_from_csv(
    request: Request,
    file: Annotated[UploadFile, File()],
    side: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    model = current_model(request)
    payload = await file.read(MAX_CSV_BYTES + 1)
    try:
        frame = parse_csv(payload)
        return predict_csv(model, frame, side=side)
    except (ValueError, OverflowError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
