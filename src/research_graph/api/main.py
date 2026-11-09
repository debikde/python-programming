"""FastAPI application for the NTNU terrain research model."""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from research_graph.api.inference import (
    BUNDLED_ARTIFACT_DIR,
    DEFAULT_MODEL_NAME,
    activate_trusted_model,
    read_registry,
)
from research_graph.api.routes import router


def create_app(
    *,
    registry_path: Path | None = None,
    artifact_dir: Path | None = None,
    default_model_name: str | None = DEFAULT_MODEL_NAME,
) -> FastAPI:
    app = FastAPI(title="Assistive Gait Research", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    registry_path = registry_path or Path(__file__).with_name("trusted_models.json")
    artifact_dir = artifact_dir or BUNDLED_ARTIFACT_DIR
    app.state.registry = read_registry(registry_path)
    app.state.artifact_dir = artifact_dir
    app.state.default_model_name = default_model_name
    app.state.models = {}
    if default_model_name is not None:
        registered = app.state.registry[default_model_name]
        model_path = artifact_dir / registered.filename
        _, model = activate_trusted_model(
            model_path.read_bytes(), app.state.registry, artifact_dir
        )
        app.state.models[default_model_name] = model
    validation_path = BUNDLED_ARTIFACT_DIR / "lab10_validation.json"
    app.state.validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.is_file()
        else None
    )
    app.include_router(router)
    return app


app = create_app()
