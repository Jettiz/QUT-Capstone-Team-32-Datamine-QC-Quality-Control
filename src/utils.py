"""
Shared Utilities Module

Small, dependency-free helpers reused across detectors.

Key Functions:
- atomic_write_csv(df, path): crash-safe CSV write (temp file + replace)
- load_csv_or_empty(path, columns, parse_dates): CSV load with a
  correctly-typed empty fallback when the file doesn't exist yet
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Write `df` to `path` as CSV without ever leaving a partially-written
    file in place: writes to a sibling `.tmp` file first, then atomically
    replaces the destination. An interrupted write leaves the previous
    version of `path` untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def load_csv_or_empty(path: Path, columns: List[str], parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load `path` as CSV, or return an empty DataFrame with `columns` (and
    correctly-typed datetime columns per `parse_dates`) if it doesn't exist
    yet -- lets a pipeline degrade gracefully on first run instead of
    erroring on a missing file.
    """
    path = Path(path)
    parse_dates = parse_dates or []

    if not path.is_file():
        empty = pd.DataFrame(columns=columns)
        for col in parse_dates:
            empty[col] = pd.to_datetime(empty[col])
        return empty

    return pd.read_csv(path, parse_dates=parse_dates)
