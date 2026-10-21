from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from research_graph.config import get_settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Isolate tests from local .env files and RG_* variables."""

    for key in list(os.environ):
        if key.upper().startswith("RG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def isolated_logging() -> Iterator[None]:
    """Keep CLI logging changes away from pytest and subsequent tests."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    noisy = [logging.getLogger(name) for name in ("httpx", "urllib3")]
    levels = [logger.level for logger in noisy]
    root.handlers.clear()
    try:
        yield
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)
        for logger, previous in zip(noisy, levels, strict=True):
            logger.setLevel(previous)
