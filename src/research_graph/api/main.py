"""FastAPI application for the NTNU terrain research model."""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from research_graph.api.inference import DEFAULT_ARTIFACT_DIR, read_registry
from research_graph.api.routes import router


def create_app(
    *, registry_path: Path | None = None, artifact_dir: Path | None = None
) -> FastAPI:
    app = FastAPI(title="Assistive Gait Research", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    registry_path = registry_path or Path(__file__).with_name("trusted_models.json")
    app.state.registry = read_registry(registry_path)
    app.state.artifact_dir = artifact_dir or DEFAULT_ARTIFACT_DIR
    app.state.model = None
    app.state.model_name = None
    validation_path = (
        Path(__file__).resolve().parents[3] / "docs" / "lab10_validation.json"
    )
    app.state.validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.is_file()
        else None
    )
    app.include_router(router)
    return app


app = create_app()
