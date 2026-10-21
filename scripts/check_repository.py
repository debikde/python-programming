from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {
        ".gitignore",
        ".env.example",
    }


def main() -> int:
    errors: list[str] = []
    for path in tracked_files():
        if path.name == ".env" or ".venv" in path.parts:
            errors.append(f"forbidden tracked file: {path}")
            continue
        if not is_text_file(path):
            continue
        content = path.read_text(encoding="utf-8")
        if content and not content.endswith("\n"):
            errors.append(f"missing final newline: {path}")
        for line_number, line in enumerate(content.splitlines(), start=1):
            if line.rstrip() != line:
                errors.append(f"trailing whitespace: {path}:{line_number}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
