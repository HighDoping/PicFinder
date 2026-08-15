#!/usr/bin/env python3
"""Regenerate PySide6 modules from Qt Designer ``.ui`` files."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

UI_DIRECTORY = Path(__file__).parent.parent


def main() -> int:
    compiler = shutil.which("pyside6-uic")
    if compiler is None:
        print(
            "pyside6-uic was not found. Run this script with the project environment.",
            file=sys.stderr,
        )
        return 1

    ui_files = sorted(UI_DIRECTORY.glob("*.ui"))
    if not ui_files:
        print(f"No .ui files found in {UI_DIRECTORY}", file=sys.stderr)
        return 1

    for ui_file in ui_files:
        output_file = ui_file.with_name(f"{ui_file.stem}_ui.py")
        print(f"{ui_file.relative_to(UI_DIRECTORY)} -> {output_file.name}")
        subprocess.run([compiler, str(ui_file), "-o", str(output_file)], check=True)
        output_file.write_text(output_file.read_text().rstrip() + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
