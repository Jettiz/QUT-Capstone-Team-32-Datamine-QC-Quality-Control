"""
Result contract between a QC method and the report/POC.

A QC method's adapter (e.g. src/qc_report/lcs_method.py) turns its
detector's output into these objects; poc/server.py serialises them with
to_dict()/to_summary() and the browser renders them without knowing which
QC method produced them. Nothing here contains analysis logic.

Shape (as JSON):

    QCMethodResult  {id, label, usable, history_mode, validation, notices,
                     counts, jobs: [QCJob summary...]}
    QCJob           {key, job, instruments, first_date, counts, n_items,
                     items: [QCItem...]}        (items only in the job view)
    QCItem          {id, code, variant, state, status_detail, reason,
                     magnitude, metrics: [{label, value}], job, instrument,
                     has_chart}

`state` is always one of src/qc_status.py's FAIL / WARNING / PASS.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import pandas as pd

from ..qc_status import FAIL, PASS, STATE_RANK, STATES, UNKNOWN_RANK, WARNING

JOB_UNKNOWN = "Job unknown"
INSTRUMENT_UNKNOWN = "Instrument unknown"

# Text forms of "no value" that must never reach the user.
_MISSING_TEXT = {"", "nan", "none", "null", "nat", "<na>"}


def is_missing(value: Any) -> bool:
    """True for None / NaN / NaT / pd.NA / blank or 'nan'/'None'-like text."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _MISSING_TEXT


def display_job(value: Any) -> str:
    """Job label for the user: the job value, or "Job unknown"."""
    return JOB_UNKNOWN if is_missing(value) else str(value).strip()


def display_instrument(value: Any) -> str:
    """Instrument label for the user: the instrument value, or "Instrument unknown"."""
    return INSTRUMENT_UNKNOWN if is_missing(value) else str(value).strip()


def display_instruments(values: Iterable[Any]) -> List[str]:
    """
    Distinct instrument labels for a set of observations (e.g. one job), in
    sorted order. "Instrument unknown" is only listed when at least one
    observation has no instrument.
    """
    labels = {display_instrument(v) for v in values}
    known = sorted(label for label in labels if label != INSTRUMENT_UNKNOWN)
    return known + ([INSTRUMENT_UNKNOWN] if INSTRUMENT_UNKNOWN in labels else [])


def count_states(items: Iterable["QCItem"]) -> Dict[str, int]:
    counts = {state: 0 for state in STATES}
    for item in items:
        if item.state in counts:
            counts[item.state] += 1
    return counts


@dataclass
class QCItem:
    """
    One analysed result the user can select, e.g. one analyte in one job.

    id            unique within its QCMethodResult (used in the chart URL)
    code          the selection key shown on the button (LCS: ANALYTE_CODE)
    variant       distinguishes several results with the same code in one job
                  (LCS: SCHEME_CODE); "" when there is only ever one
    state         FAIL / WARNING / PASS
    status_detail the method's own, finer outcome (LCS: drift_status)
    reason        plain-language explanation of the state
    magnitude     tie-break within a state; larger is shown first
    metrics       ordered [{label, value}] -- values already formatted text
    job / instrument  display labels (never None/NaN -- use display_*())
    has_chart     whether render_chart() can draw this item
    chart_payload whatever the method needs to draw the chart later; kept
                  server-side, never serialised
    """
    id: str
    code: str
    variant: str
    state: str
    status_detail: str
    reason: str
    magnitude: float
    metrics: List[Dict[str, str]]
    job: str
    instrument: str
    has_chart: bool = True
    chart_payload: Any = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "variant": self.variant,
            "state": self.state,
            "status_detail": self.status_detail,
            "reason": self.reason,
            "magnitude": float(self.magnitude) if pd.notna(self.magnitude) else 0.0,
            "metrics": self.metrics,
            "job": self.job,
            "instrument": self.instrument,
            "has_chart": self.has_chart,
        }


@dataclass
class QCJob:
    """All items of one job (the unit Step 2 lists and Step 3 shows)."""
    key: str
    job: str
    instruments: List[str]
    first_date: str
    items: List[QCItem]

    def counts(self) -> Dict[str, int]:
        return count_states(self.items)

    def sorted_items(self) -> List[QCItem]:
        return sorted(self.items, key=lambda i: (STATE_RANK.get(i.state, UNKNOWN_RANK), -abs(i.magnitude or 0.0)))

    def to_summary(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "job": self.job,
            "instruments": self.instruments,
            "first_date": self.first_date,
            "counts": self.counts(),
            "n_items": len(self.items),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self.to_summary(), "items": [i.to_dict() for i in self.sorted_items()]}


@dataclass
class QCMethodResult:
    """Everything one QC method produced for one uploaded file."""
    id: str
    label: str
    usable: bool
    history_mode: str
    validation: Dict[str, Any]
    notices: List[str]
    jobs: List[QCJob]

    def counts(self) -> Dict[str, int]:
        return count_states(item for job in self.jobs for item in job.items)

    def find_item(self, item_id: str):
        for job in self.jobs:
            for item in job.items:
                if item.id == item_id:
                    return item
        return None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "usable": self.usable,
            "history_mode": self.history_mode,
            "validation": self.validation,
            "notices": self.notices,
            "counts": self.counts(),
            "jobs": [job.to_summary() for job in self.jobs],
        }


__all__ = [
    "FAIL", "WARNING", "PASS", "JOB_UNKNOWN", "INSTRUMENT_UNKNOWN",
    "is_missing", "display_job", "display_instrument", "display_instruments", "count_states",
    "QCItem", "QCJob", "QCMethodResult",
]
