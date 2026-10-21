from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from research_graph.config import Settings, get_settings


def test_settings_defaults() -> None:
    settings = get_settings()

    assert settings.github_token is None
    assert settings.data_dir == Path("data")
    assert settings.log_level == "INFO"
    assert settings.request_timeout == 10.0
    assert settings.max_concurrency == 4


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RG_LOG_LEVEL", "debug")
    monkeypatch.setenv("RG_MAX_CONCURRENCY", "16")
    monkeypatch.setenv("RG_DATA_DIR", "custom/data")
    monkeypatch.setenv("RG_REQUEST_TIMEOUT", "25.5")
    monkeypatch.setenv("RG_GITHUB_TOKEN", "synthetic_test_token_do_not_use")

    settings = Settings()

    assert settings.log_level == "DEBUG"
    assert settings.max_concurrency == 16
    assert settings.data_dir == Path("custom/data")
    assert settings.request_timeout == 25.5
    assert settings.github_token == "synthetic_test_token_do_not_use"
    assert settings.github_token not in repr(settings)


@pytest.mark.parametrize("value", ["0", "-1", "65", "1.5", "invalid"])
def test_rejects_invalid_concurrency(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("RG_MAX_CONCURRENCY", value)

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("value", ["0", "-1", "0.09", "301", "nan", "inf", "invalid"])
def test_rejects_invalid_timeout(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RG_REQUEST_TIMEOUT", value)
    with pytest.raises(ValidationError):
        Settings()


def test_rejects_invalid_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RG_LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError):
        Settings()


def test_ensure_data_dir_creates_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "nested" / "dataset"

    settings = Settings(data_dir=data_dir)
    settings.ensure_data_dir()

    assert data_dir.is_dir()
    marker = data_dir / "existing.txt"
    marker.write_text("keep", encoding="utf-8")
    settings.ensure_data_dir()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_data_directory_does_not_overwrite_files(tmp_path: Path) -> None:
    occupied = tmp_path / "file"
    occupied.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        Settings(data_dir=occupied).ensure_data_dir()
    assert occupied.read_text(encoding="utf-8") == "keep"


def test_dotenv_and_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "RG_REQUEST_TIMEOUT=20\nRG_MAX_CONCURRENCY=8\nRG_GITHUB_TOKEN=\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RG_MAX_CONCURRENCY", "2")
    settings = get_settings()
    assert settings.request_timeout == 20
    assert settings.max_concurrency == 2
    assert settings.github_token is None
