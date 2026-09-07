"""Repo-root pytest conftest: makes `src` importable as a namespace package
regardless of the directory pytest is invoked from (no setup.py/pyproject.toml
package config exists in this project)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
