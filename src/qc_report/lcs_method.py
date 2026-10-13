"""
LCS (Control) adapter for the report/POC -- the reference implementation of
a QC method plugging into src/qc_report (see docs/QC_INTEGRATION_GUIDE.md).

It contains NO analysis logic: routing comes from
data_validator.select_lcs_rows(), validation from validate_lcs_data(), the
analysis and the FAIL/WARNING/PASS decision from LCSDetector, and the chart
from control_visualize.plot_evaluation(). This file only:
  1. decides which history to analyse against (the POC's History toggle),
  2. maps each LCS evaluation onto a QCItem (formatted, user-facing text),
  3. groups items per job.

History toggle (demo behaviour):
  - on:  the upload's LCS jobs are ordered by date; the older half seeds the
         history store (rebuilt on every run, written to the config's
         demo_history_path) and the newer half is analysed job by job.
  - off: every job is analysed in date order against an in-memory history
         built from earlier jobs in the same file; nothing is stored.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from ..data_validator import select_lcs_rows, validate_lcs_data
from ..detectors.control_detector import LCSDetector
from ..visualisers.control_visualize import plot_evaluation
from .contract import (
    QCItem, QCJob, QCMethodResult, display_instrument, display_instruments, display_job, is_missing,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "lcs_config.yaml"

# Fraction of the upload's jobs (oldest first) used as history when the
# POC's History toggle is on.
HISTORY_SEED_FRACTION = 0.5

_UNIT_LABELS = {"MG_KG": "mg/kg", "PPM": "ppm", "PPB": "ppb", "PERC": "%", "PCT": "%", "G_T": "g/t"}

_POINT_STATUS_TEXT = {
    "UPPER_FAILURE": "Above the upper failure limit",
    "LOWER_FAILURE": "Below the lower failure limit",
    "UPPER_WARNING": "In the upper warning band",
    "LOWER_WARNING": "In the lower warning band",
    "NORMAL": "Within the warning limits",
}

_DROP_REASONS = {
    "n_dropped_missing_analyte_or_scheme": "missing analyte or scheme code",
    "n_dropped_missing_date": "missing analysed date",
    "n_dropped_missing_value": "missing result value",
    "n_dropped_missing_limits": "missing target or limit values",
    "n_dropped_zero_span": "target equal to a failure limit",
    "n_dropped_target_outside_range": "target outside its own failure limits",
}


def _num(value: Any, fmt: str = ".4g") -> Optional[str]:
    if is_missing(value):
        return None
    return format(float(value), fmt)


def _unit(unit_code: Any) -> str:
    if is_missing(unit_code):
        return ""
    return _UNIT_LABELS.get(str(unit_code).strip().upper(), str(unit_code).strip())


def _with_unit(value: Optional[str], unit: str) -> Optional[str]:
    if value is None:
        return None
    return f"{value} {unit}".strip()


def _date(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def _job_key(series: pd.Series) -> pd.Series:
    """Same non-null job identifier LCSDetector orders/de-duplicates by."""
    return series.fillna("").astype(str)


class LCSMethod:
    """QC method adapter for Control (LCS). Registered in registry.py."""

    id = "control"          # must match the detector id used by the POC (poc/data.js)
    label = "Control (LCS)"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        demo_path = Path(config.get("demo_history_path", "data/processed/lcs_history_poc.csv"))
        self.demo_history_path = demo_path if demo_path.is_absolute() else REPO_ROOT / demo_path

    # ── QCMethod interface ───────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Real validation (data_validator.validate_lcs_data) + a usable flag."""
        report = validate_lcs_data(df)
        usable = not report["missing_columns"] and report["n_rows"] > 0
        notes = []
        if report["missing_columns"]:
            notes.append("Missing columns: " + ", ".join(report["missing_columns"]))
        if report["n_rows"] == 0 and not report["missing_columns"]:
            notes.append("No LCS rows found (Standard rows with STD_LOT_CODE 'Sample').")
        return {
            "usable": usable,
            "n_rows": int(report["n_rows"]),
            "missing_columns": list(report["missing_columns"]),
            "null_counts": {k: int(v) for k, v in report["null_counts"].items()},
            "dtype_issues": dict(report["dtype_issues"]),
            "notes": notes,
        }

    def run(self, df: pd.DataFrame, *, use_history: bool) -> QCMethodResult:
        history_mode = "on" if use_history else "off"
        validation = self.validate(df)
        if not validation["usable"]:
            return QCMethodResult(self.id, self.label, False, history_mode, validation,
                                  validation["notes"] or ["LCS analysis cannot run on this file."], [])

        lcs = select_lcs_rows(df).copy()
        lcs["_job_key"] = _job_key(lcs["job_code"]) if "job_code" in lcs.columns else ""
        dated = lcs[lcs["analysed_date"].notna()]
        job_first_date = dated.groupby("_job_key")["analysed_date"].min().sort_values(kind="stable")
        jobs_oldest_first = job_first_date.index.tolist()

        notices: List[str] = []
        if use_history:
            detector = LCSDetector(str(self.config_path), history_path=str(self.demo_history_path))
            n_seed = int(len(jobs_oldest_first) * HISTORY_SEED_FRACTION)
            seed_jobs = set(jobs_oldest_first[:n_seed])
            seed = detector.build_history(lcs[lcs["_job_key"].isin(seed_jobs)])
            to_analyse = lcs[~lcs["_job_key"].isin(seed_jobs)]
            result = detector.detect(to_analyse, history=seed, persist=True)
            if n_seed == 0:
                notices.append("History on, but this file contains only one LCS job, so there is no older half "
                               "to use as history. It was analysed without history.")
            else:
                notices.append(f"History on: the older {n_seed} of {len(jobs_oldest_first)} LCS jobs in this file "
                               f"were used as history ({len(seed)} stored observations); the newer "
                               f"{len(jobs_oldest_first) - n_seed} job(s) were analysed against it.")
        else:
            detector = LCSDetector(str(self.config_path))
            result = detector.detect(lcs, history=pd.DataFrame(), persist=False)
            notices.append("History off: jobs were analysed in date order using only earlier jobs in this file; "
                           "nothing was stored.")

        report = result.details.get("validation_report", {})
        dropped = [f"{report[k]:,} with {text}" for k, text in _DROP_REASONS.items() if report.get(k)]
        if dropped:
            notices.append("Some LCS rows were skipped: " + "; ".join(dropped) + ".")
        validation["detector_report"] = {k: int(v) for k, v in report.items()}

        jobs = self._build_jobs(result.details.get("results", []), lcs, job_first_date)
        return QCMethodResult(self.id, self.label, True, history_mode, validation, notices, jobs)

    def render_chart(self, item: QCItem, output_path: Path) -> Path:
        return plot_evaluation(item.chart_payload, output_path)

    # ── Mapping ──────────────────────────────────────────────────────────────

    def _build_jobs(self, evaluations: List[Dict[str, Any]], lcs: pd.DataFrame,
                    job_first_date: pd.Series) -> List[QCJob]:
        by_job: Dict[str, List[Dict[str, Any]]] = {}
        for evaluation in evaluations:  # already in job order
            key = "" if is_missing(evaluation.get("job_code")) else str(evaluation["job_code"])
            by_job.setdefault(key, []).append(evaluation)

        instruments_col = lcs["instrument_id"] if "instrument_id" in lcs.columns else pd.Series(pd.NA, index=lcs.index)
        jobs: List[QCJob] = []
        item_counter = 0
        for job_index, (key, job_evals) in enumerate(by_job.items()):
            job_rows = lcs["_job_key"] == key
            # Two results for the same analyte in one job (different scheme or
            # standard) need a variant label to tell them apart.
            seen_schemes: Dict[tuple, set] = {}
            for e in job_evals:
                seen_schemes.setdefault((e["analyte_code"], e["scheme_code"]), set()).add(e["std_code"])

            items = []
            for e in job_evals:
                variant = str(e["scheme_code"])
                if len(seen_schemes[(e["analyte_code"], e["scheme_code"])]) > 1:
                    variant = f"{e['std_code']} / {e['scheme_code']}"
                items.append(self._to_item(str(item_counter), e, variant))
                item_counter += 1

            first = job_first_date.get(key)
            jobs.append(QCJob(
                key=str(job_index),
                job=display_job(key),
                instruments=display_instruments(instruments_col[job_rows]),
                first_date=_date(first) or "",
                items=items,
            ))
        return jobs

    def _to_item(self, item_id: str, e: Dict[str, Any], variant: str) -> QCItem:
        unit = _unit(e.get("unit_code"))
        metrics: List[Dict[str, str]] = []

        def add(label: str, value: Optional[str]):
            if value is not None and value != "":
                metrics.append({"label": label, "value": value})

        add("Result", _with_unit(_num(e["measured_value"]), unit))
        add("Target", _with_unit(_num(e["target_value"]), unit))
        add("Warning limits", _with_unit(f"{_num(e['limit_min_warning'])} – {_num(e['limit_max_warning'])}", unit))
        add("Failure limits", _with_unit(f"{_num(e['limit_min'])} – {_num(e['limit_max'])}", unit))
        add("Offset", f"{e['offset']:+.3f} (target = 0, failure limits = ±1)")
        add("Point check", _POINT_STATUS_TEXT.get(e["point_status"], e["point_status"]))

        if e["drift_status"] == "INSUFFICIENT_HISTORY" or e["n_history"] < 3:
            add("Trend", "Not assessed (fewer than 3 observations)")
        elif not is_missing(e.get("trend_slope")):
            add("Trend", f"{e['trend_direction'].capitalize()}, slope {e['trend_slope']:+.4f} per result, "
                         f"consistency {e['trend_strength']:.2f} (last {e['trend_n_used']} results)")
        add("Observations used", f"{e['n_history']} (since {_date(e.get('oldest_date')) or 'n/a'})")

        # The projection is an unsigned distance / slope, so it is only
        # meaningful while the result is still inside its warning limits.
        est_warn, est_fail = _num(e.get("est_obs_to_warning"), ".1f"), _num(e.get("est_obs_to_failure"), ".1f")
        if e["point_status"] == "NORMAL" and (est_warn or est_fail):
            parts = []
            if est_warn:
                parts.append(f"~{est_warn} to the warning limit")
            if est_fail:
                parts.append(f"~{est_fail} to the failure limit")
            add("Estimated results to limit", ", ".join(parts))

        add("Company status (STANDARD_STATUS)",
            "Not recorded" if is_missing(e.get("standard_status")) else str(e["standard_status"]))
        std = str(e["std_code"])
        if not is_missing(e.get("std_lot_code")):
            std += f" (lot {e['std_lot_code']})"
        add("Standard / scheme", f"{std} / {e['scheme_code']}")
        add("Analysed", _date(e.get("analysed_date")))
        add("Instrument", display_instrument(e.get("instrument_id")))
        add("Kept in history", "Yes" if e["eligible_for_history"] else
            "No: offset is beyond ±2, so it is flagged but not used for future trends")
        if e.get("data_quality_flag"):
            add("Data quality", str(e["data_quality_flag"]).capitalize())

        return QCItem(
            id=item_id,
            code=str(e["analyte_code"]),
            variant=variant,
            state=e["warning_state"],
            status_detail=e["drift_status"],
            reason=e["drift_reason"],
            magnitude=abs(float(e["offset"])),
            metrics=metrics,
            job=display_job(e.get("job_code")),
            instrument=display_instrument(e.get("instrument_id")),
            has_chart=True,
            chart_payload=e,
        )
