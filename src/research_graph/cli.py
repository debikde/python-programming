"""Command line interface for Assistive Gait Research."""

from __future__ import annotations

import typer
from pydantic import ValidationError
from pydantic_settings import SettingsError

from research_graph import __version__
from research_graph.config import Settings, get_settings
from research_graph.logging_setup import setup_logging

app = typer.Typer(
    help="Assistive Gait Research: project configuration utilities.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def format_settings(settings: Settings) -> list[str]:
    """Format settings for safe CLI output."""

    token_present = "true" if settings.github_token else "false"
    lines = [
        f"data_dir={settings.data_dir}",
        f"log_level={settings.log_level}",
        f"request_timeout={settings.request_timeout}",
        f"max_concurrency={settings.max_concurrency}",
        f"github_token_present={token_present}",
    ]
    # Redact accidental occurrences in other user-configurable fields too.
    if settings.github_token:
        return [line.replace(settings.github_token, "[REDACTED]") for line in lines]
    return lines


@app.command()
def version() -> None:
    """Print project version."""

    typer.echo(__version__)


@app.command("check-config")
def check_config() -> None:
    """Load settings, configure logging, and print a safe summary."""

    try:
        settings = get_settings()
    except (ValidationError, SettingsError):
        typer.echo("Invalid configuration. Check RG_* values and .env.", err=True)
        raise typer.Exit(code=2) from None
    setup_logging(settings.log_level)
    try:
        settings.ensure_data_dir()
    except OSError:
        typer.echo(
            "Cannot create data directory. Check RG_DATA_DIR and permissions.", err=True
        )
        raise typer.Exit(code=2) from None
    for line in format_settings(settings):
        typer.echo(line)
