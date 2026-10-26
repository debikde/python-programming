"""Where the local NTNU data and research reports are stored."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT.parent / "research-publication-data"
NTNU_DIR = DATA_ROOT / "data" / "raw" / "ntnu"
REPORTS_DIR = DATA_ROOT / "artifacts" / "reports"


def report_dir(name: str) -> Path:
    return REPORTS_DIR / name
