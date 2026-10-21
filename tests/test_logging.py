"""Behavioral checks for stderr logging and repeated initialization."""

import logging
import re

import pytest

from research_graph.logging_setup import setup_logging


def test_logging_format_and_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    logging.getLogger("research_graph.test").info("ready")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO research_graph.test: ready\n",
        captured.err,
    )


def test_reinitialization_does_not_duplicate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging("INFO")
    first = logging.getLogger().handlers[0]
    setup_logging("DEBUG")
    assert len(logging.getLogger().handlers) == 1
    assert first not in logging.getLogger().handlers
    logging.getLogger("research_graph.test").debug("one message")
    assert capsys.readouterr().err.count("one message") == 1


def test_logging_filters_noise(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("DEBUG")
    logging.getLogger("httpx").info("quiet")
    logging.getLogger("urllib3").debug("quiet")
    logging.getLogger("httpx").warning("visible")
    captured = capsys.readouterr()
    assert "quiet" not in captured.err
    assert "WARNING httpx: visible" in captured.err


def test_logging_respects_level(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("ERROR")
    logger = logging.getLogger("research_graph.test")
    logger.warning("quiet")
    logger.error("visible")
    captured = capsys.readouterr()
    assert "quiet" not in captured.err
    assert "ERROR research_graph.test: visible" in captured.err
