from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_graph.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_check_config_outputs_safe_summary(tmp_path: Path) -> None:
    assert not (tmp_path / ".env").exists()
    result = runner.invoke(app, ["check-config"])

    assert result.exit_code == 0
    assert "data_dir=data" in result.stdout
    assert "github_token_present=false" in result.stdout
    assert (tmp_path / "data").is_dir()


@pytest.mark.parametrize("token", ["short", "synthetic_test_token_do_not_use"])
def test_check_config_hides_token(monkeypatch: pytest.MonkeyPatch, token: str) -> None:
    monkeypatch.setenv("RG_GITHUB_TOKEN", token)

    result = runner.invoke(app, ["check-config"])

    assert result.exit_code == 0
    assert token not in result.output
    assert token not in result.stderr
    assert "github_token_present=true" in result.stdout
    assert "github_token=" not in result.stdout


def test_help_lists_only_lab1_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.stdout
    assert "check-config" in result.stdout


@pytest.mark.parametrize(
    "field", ["RG_LOG_LEVEL", "RG_REQUEST_TIMEOUT", "RG_MAX_CONCURRENCY"]
)
def test_invalid_config_is_safe(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    token = "synthetic_test_token_do_not_use"
    monkeypatch.setenv("RG_GITHUB_TOKEN", token)
    monkeypatch.setenv(field, token)
    result = runner.invoke(app, ["check-config"])
    assert result.exit_code == 2
    assert "Invalid configuration" in result.stderr
    assert token not in result.output


def test_data_directory_failure_is_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token = "synthetic_test_token_do_not_use"
    occupied = tmp_path / token
    occupied.write_text("existing file", encoding="utf-8")
    monkeypatch.setenv("RG_GITHUB_TOKEN", token)
    monkeypatch.setenv("RG_DATA_DIR", str(occupied))
    result = runner.invoke(app, ["check-config"])
    assert result.exit_code == 2
    assert "Cannot create data directory" in result.stderr
    assert token not in result.output
    assert occupied.read_text(encoding="utf-8") == "existing file"


def test_token_in_other_settings_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "synthetic_test_token_do_not_use"
    monkeypatch.setenv("RG_GITHUB_TOKEN", token)
    monkeypatch.setenv("RG_DATA_DIR", token)
    result = runner.invoke(app, ["check-config"])
    assert result.exit_code == 0
    assert token not in result.output
    assert "[REDACTED]" in result.stdout


def test_check_config_reads_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("RG_MAX_CONCURRENCY=7\n", encoding="utf-8")
    result = runner.invoke(app, ["check-config"])
    assert result.exit_code == 0
    assert "max_concurrency=7" in result.stdout
