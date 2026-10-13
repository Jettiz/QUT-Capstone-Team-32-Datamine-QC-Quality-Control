"""
Registry of QC methods the report/POC can run.

poc/server.py runs every method registered here on each uploaded file and
serves the results; the browser shows real results for these methods and
keeps its placeholder data for every other detector card.

To add a QC method (see docs/QC_INTEGRATION_GUIDE.md):
  1. write an adapter class that satisfies `QCMethod` below (copy
     lcs_method.py's structure), with `id` equal to the detector id the POC
     already uses for its card in poc/data.js;
  2. add one `register(YourMethod())` line at the bottom of this file.
Nothing else in the server or the browser needs to change.
"""

from pathlib import Path
from typing import Any, Dict, Protocol

import pandas as pd

from .contract import QCItem, QCMethodResult
from .lcs_method import LCSMethod


class QCMethod(Protocol):
    """What the server needs from a QC method adapter."""

    id: str      # POC detector id, e.g. "control"
    label: str   # display name, e.g. "Control (LCS)"

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Can this method run on the loaded (internal-schema) data? Must return
        at least {"usable": bool, "n_rows": int, "missing_columns": [...],
        "notes": [str, ...]}.
        """
        ...

    def run(self, df: pd.DataFrame, *, use_history: bool) -> QCMethodResult:
        """
        Validate, select this method's rows, analyse, and return the result.
        Must not raise for 'cannot run' situations -- return usable=False
        with notices instead.
        """
        ...

    def render_chart(self, item: QCItem, output_path: Path) -> Path:
        """Draw `item`'s chart as a PNG at output_path (called on demand)."""
        ...


QC_METHODS: Dict[str, QCMethod] = {}


def register(method: QCMethod) -> None:
    if method.id in QC_METHODS:
        raise ValueError(f"QC method id {method.id!r} is already registered")
    QC_METHODS[method.id] = method


def get_method(method_id: str) -> QCMethod:
    return QC_METHODS[method_id]


# ── Registered methods ───────────────────────────────────────────────────────
register(LCSMethod())
