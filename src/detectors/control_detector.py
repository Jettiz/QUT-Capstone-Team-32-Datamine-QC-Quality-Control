"""
LCS (Laboratory Control Sample) Historic Drift Detector

Detects deviations in control samples that verify method accuracy and
stability, tracked **across jobs over time** via a per-analyte rolling
history -- not just within a single analytical run.

Behavioural reference: notebooks/LCS_drift_detection_historic.ipynb (see
that notebook and its design notes for the full rationale).

Input
-----
The INTERNAL column schema produced by src/data_loader.load_qc_data()
(analytical_type, std_lot_code, std_code, scheme_code, job_code,
analyte_code, analysed_date, measured_value, target_value, limit_*, ...).
Rows are routed to LCS by src/data_validator.select_lcs_rows() (Standard
rows whose std_lot_code is "Sample") -- this module never re-implements that
split.

Method (as in the notebook)
---------------------------
- Independent history per (analyte_code, std_code, scheme_code) group,
  capped at `max_history_per_analyte` (default 30) observations.
- Each observation is scaled against its own limits so target -> 0,
  limit_max -> +1, limit_min -> -1 (piecewise: divide by the upper span
  above target, the lower span below it). This scaled value is the
  "offset" (transformed_value).
- Jobs are processed in chronological order (a job's time = its earliest
  analysed_date). For each job and each group it touches: the job's rows are
  appended to that group's history, observations are de-duplicated on
  DEDUP_KEYS (so re-running the same export never double-counts), the
  history is trimmed to the cap, and the group is evaluated ONCE, on the
  job's latest observation.
- A Theil-Sen slope (robust to a single outlier, unlike OLS) over the most
  recent `trend_window` observations, with x = equal-step observation order
  (not calendar time), gives a trend direction/strength.
- Two axes are combined into one drift_status: has the latest result already
  breached a limit (_classify_point_status), and/or is a consistent trend
  heading toward one (_classify_drift).

Deliberate additions on top of the notebook
--------------------------------------------
- Offset-exclusion rule: an observation is always analysed and flagged, but
  is only kept in the stored history if |offset| <= `offset_exclusion_bound`
  (default 2.0, twice the target-to-failure span), so a one-off wild reading
  can't distort future trend fits. Exactly +/- the bound is still eligible.
- Point-in-time guard: when a group is evaluated for a job, stored
  observations dated AFTER that job's latest observation are ignored, so an
  older upload is never compared against "future" results.
- Too little history (n < min_history_point) no longer hides a limit
  breach: the latest result is still checked against its limits; only the
  trend assessment is skipped.
- Each evaluation is mapped to one of the three shared QC warning states
  (src/qc_status.py) via WARNING_STATE_BY_DRIFT_STATUS below.

Configuration: config/lcs_config.yaml
- history_path: where the stored rolling history CSV lives
- excluded_std_codes: std_code values that aren't genuine reference materials
- max_history_per_analyte: rolling-history cap per group (default 30)
- min_history_point / min_history_trend: two-tier minimum-data thresholds
- trend_window, slope_eps, trend_strength_min_for_escalation,
  trend_progress_min_for_failure_drift: trend/classification tuning
- offset_exclusion_bound: the history-eligibility cutoff
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import theilslopes

from .base_detector import LCSAnomalyResult
from ..data_validator import select_lcs_rows
from ..qc_status import FAIL, PASS, STATE_RANK, WARNING
from ..utils import atomic_write_csv, load_csv_or_empty

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ── Grouping ─────────────────────────────────────────────────────────────────
# One independent historic time series = one (analyte_code, std_code,
# scheme_code) combination -- see notebooks/LCS_drift_detection_historic.ipynb
# for why std_code (not std_lot_code) and scheme_code are the identity fields,
# and why job_code is deliberately NOT part of the grouping key (it is the
# arrival unit instead).
GROUP_KEYS = ["analyte_code", "std_code", "scheme_code"]

# Business key for de-duplicating observations across repeated runs.
DEDUP_KEYS = GROUP_KEYS + ["job_code", "analysed_date", "measured_value"]

# Columns that must exist on the input frame (rows with missing values in the
# date/limit columns are dropped and reported, not the whole run failing --
# see _validate_lcs_frame()). job_code / instrument_id are optional.
REQUIRED_COLUMNS = [
    "analytical_type", "std_lot_code", "std_code", "scheme_code", "analyte_code",
    "analysed_date", "measured_value",
    "target_value", "limit_max", "limit_min", "limit_max_warning", "limit_min_warning",
]

# Numeric/date columns coerced to their proper dtype on ingestion, in case the
# caller passes an untyped frame.
_NUMERIC_COLUMNS = [
    "measured_value", "target_value", "limit_max", "limit_min", "limit_max_warning", "limit_min_warning",
]

# Columns kept in the stored history -- raw traceability columns plus the
# transformed value/limits computed at ingestion time (raw for display/audit,
# transformed to avoid recomputing against possibly-since-changed limits).
HISTORY_COLUMNS = [
    "analyte_code", "std_code", "std_lot_code", "scheme_code", "job_code", "instrument_id",
    "analysed_date", "unit_code", "standard_status",
    "measured_value", "target_value", "limit_max", "limit_min",
    "limit_max_warning", "limit_min_warning",
    "limit_max_inclusive", "limit_min_inclusive",
    "limit_max_warning_inclusive", "limit_min_warning_inclusive",
    "transformed_value", "transformed_target", "transformed_max", "transformed_min",
    "transformed_max_warning", "transformed_min_warning",
]

# ── Warning states ───────────────────────────────────────────────────────────
# LCS-specific mapping from the detector's analysis outcome (drift_status) to
# the three shared QC states. The finer distinction (e.g. a *_FAILURE_DRIFT
# being further along than a *_WARNING_DRIFT) stays visible through
# drift_status and drift_reason.
WARNING_STATE_BY_DRIFT_STATUS = {
    "UPPER_FAILURE": FAIL,
    "LOWER_FAILURE": FAIL,
    "UPPER_WARNING": WARNING,
    "LOWER_WARNING": WARNING,
    "UPPER_WARNING_DRIFT": WARNING,
    "LOWER_WARNING_DRIFT": WARNING,
    "UPPER_FAILURE_DRIFT": WARNING,
    "LOWER_FAILURE_DRIFT": WARNING,
    "NORMAL": PASS,
    "INSUFFICIENT_HISTORY": PASS,
}

# AnomalyResult.severity vocabulary ("low/medium/high/critical", see
# base_detector.py) and a fixed per-state confidence -- not a calibrated
# probability, the same convention this detector has always used.
_STATE_TO_RESULT_SEVERITY = {FAIL: "critical", WARNING: "medium", PASS: "low"}
_STATE_CONFIDENCE = {FAIL: 100.0, WARNING: 75.0, PASS: 95.0}

# Internal helper column holding a non-null job identifier for ordering and
# de-duplication (job_code itself is left untouched, missing stays missing).
_JOB_KEY = "_job_key"


def lcs_warning_state(drift_status: str) -> str:
    """Map an LCS drift_status to FAIL / WARNING / PASS."""
    return WARNING_STATE_BY_DRIFT_STATUS.get(drift_status, WARNING)


def _ensure_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return df restricted/reordered to HISTORY_COLUMNS, adding any missing ones as NA."""
    df = df.copy()
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[HISTORY_COLUMNS]


def _filter_excluded_std_codes(df: pd.DataFrame, excluded_std_codes) -> pd.DataFrame:
    """Drop rows whose std_code is not a genuine reference-material identifier."""
    std_code = df["std_code"].fillna("").astype(str).str.strip()
    return df.loc[~std_code.isin(excluded_std_codes)].copy()


def _validate_lcs_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validate LCS rows ahead of transformation.

    Checks REQUIRED_COLUMNS are present (raises if not), then drops rows with
    a missing analyte_code/scheme_code (no group to belong to), a missing
    analysed_date, a missing measured_value, a missing target or
    any limit value, a zero-span limit (target == max or target == min --
    would divide by zero), or a target outside [limit_min, limit_max]
    (data-entry error).

    Returns (clean_df, report) where `report` counts what was dropped and
    why, so data-quality issues stay visible rather than silently vanishing.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    report = {"n_input": len(df)}
    clean = df.copy()

    missing_group_mask = clean[["analyte_code", "scheme_code"]].isna().any(axis=1)
    report["n_dropped_missing_analyte_or_scheme"] = int(missing_group_mask.sum())
    clean = clean.loc[~missing_group_mask]

    missing_date_mask = clean["analysed_date"].isna()
    report["n_dropped_missing_date"] = int(missing_date_mask.sum())
    clean = clean.loc[~missing_date_mask]

    missing_value_mask = clean["measured_value"].isna()
    report["n_dropped_missing_value"] = int(missing_value_mask.sum())
    clean = clean.loc[~missing_value_mask]

    limit_cols = ["target_value", "limit_max", "limit_min", "limit_max_warning", "limit_min_warning"]
    missing_limits_mask = clean[limit_cols].isna().any(axis=1)
    report["n_dropped_missing_limits"] = int(missing_limits_mask.sum())
    clean = clean.loc[~missing_limits_mask]

    zero_span_mask = (
        (clean["limit_max"] == clean["target_value"]) |
        (clean["target_value"] == clean["limit_min"])
    )
    report["n_dropped_zero_span"] = int(zero_span_mask.sum())
    clean = clean.loc[~zero_span_mask]

    target_outside_mask = (
        (clean["target_value"] > clean["limit_max"]) |
        (clean["target_value"] < clean["limit_min"])
    )
    report["n_dropped_target_outside_range"] = int(target_outside_mask.sum())
    clean = clean.loc[~target_outside_mask].copy()

    report["n_output"] = len(clean)
    return clean, report


def _transform_lcs_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the piecewise LCS transform: target_value -> 0, limit_max -> +1,
    limit_min -> -1, computed per row using that row's OWN target/limits. A
    value on or above target is scaled by the upper span (max - target); a
    value below target is scaled by the lower span (target - min).
    transformed_value is what the rest of this module calls "offset".

    Assumes `df` has already passed _validate_lcs_frame().
    """
    out = df.copy()
    target = out["target_value"]
    upper_span = out["limit_max"] - target
    lower_span = target - out["limit_min"]

    value = out["measured_value"]
    span_for_value = np.where(value >= target, upper_span, lower_span)
    out["transformed_value"] = (value - target) / span_for_value

    out["transformed_target"] = 0.0
    out["transformed_max"] = 1.0
    out["transformed_min"] = -1.0
    out["transformed_max_warning"] = (out["limit_max_warning"] - target) / upper_span
    out["transformed_min_warning"] = (out["limit_min_warning"] - target) / lower_span

    return out


def _calculate_trend(offsets: Sequence[float], window: int, slope_eps: float) -> Dict[str, Any]:
    """
    Theil-Sen slope over the most recent `window` offsets of one group's
    chronologically ordered history, x = equal-step observation order (not
    elapsed time). Theil-Sen (median of pairwise slopes) is used instead of
    OLS because it is resistant to a single isolated outlier tilting the fit.

    x positions are the observations' positions in the full ordered history
    (0..n-1), so `intercept` describes the trend line in those coordinates
    (used for plotting).
    """
    y_all = np.asarray(offsets, dtype=float)
    n_total = len(y_all)
    start = max(0, n_total - window)
    y = y_all[start:]
    n = len(y)

    if n < 2:
        return {"slope": np.nan, "intercept": np.nan, "direction": "FLAT", "strength": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "n_used": n}

    x = np.arange(start, n_total, dtype=float)
    slope, intercept, lo, hi = theilslopes(y, x)

    if slope > slope_eps:
        direction = "UP"
    elif slope < -slope_eps:
        direction = "DOWN"
    else:
        direction = "FLAT"

    # Consistency: fraction of consecutive, non-tied observation-to-observation
    # moves that agree in sign with the fitted slope -- separates "one outlier
    # tilted the fit" from "consistent march toward a boundary".
    diffs = np.diff(y)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0 or slope == 0:
        strength = 0.0
    else:
        strength = float((np.sign(nonzero) == np.sign(slope)).mean())

    return {"slope": float(slope), "intercept": float(intercept), "direction": direction,
            "strength": strength, "ci_low": float(lo), "ci_high": float(hi), "n_used": n}


def _calculate_threshold_distance(latest_row, trend: Dict[str, Any],
                                   trend_strength_min_for_escalation: float) -> Dict[str, Any]:
    """
    Signed-magnitude distance from the latest transformed value to the
    relevant warning/failure boundary given the trend direction, plus an
    advisory estimated-observations-to-boundary projection when the trend
    is consistent enough to trust.
    """
    value = latest_row["transformed_value"]
    direction = trend.get("direction")

    if direction == "UP":
        warning_bound, failure_bound = latest_row["transformed_max_warning"], latest_row["transformed_max"]
    elif direction == "DOWN":
        warning_bound, failure_bound = latest_row["transformed_min_warning"], latest_row["transformed_min"]
    else:
        warning_bound = failure_bound = np.nan

    distance_to_warning = abs(warning_bound - value) if pd.notna(warning_bound) else np.nan
    distance_to_failure = abs(failure_bound - value) if pd.notna(failure_bound) else np.nan
    progress_to_warning = (
        1.0 - (distance_to_warning / abs(warning_bound))
        if pd.notna(warning_bound) and warning_bound != 0 else np.nan
    )

    est_obs_to_warning = np.nan
    est_obs_to_failure = np.nan

    slope, strength = trend.get("slope"), trend.get("strength")
    is_consistent = (
        direction not in (None, "FLAT")
        and pd.notna(slope) and slope != 0
        and pd.notna(strength) and strength >= trend_strength_min_for_escalation
    )
    if is_consistent:
        per_obs_change = abs(slope)
        if pd.notna(distance_to_warning):
            est_obs_to_warning = distance_to_warning / per_obs_change
        if pd.notna(distance_to_failure):
            est_obs_to_failure = distance_to_failure / per_obs_change

    return {
        "distance_to_warning": distance_to_warning,
        "distance_to_failure": distance_to_failure,
        "progress_to_warning": progress_to_warning,
        "est_obs_to_warning": est_obs_to_warning,
        "est_obs_to_failure": est_obs_to_failure,
    }


def _is_inclusive(row, col: str) -> bool:
    """Missing inclusivity flags default to inclusive."""
    val = row.get(col)
    if pd.isna(val):
        return True
    return str(val).strip().upper() == "Y"


def _classify_point_status(latest_row) -> str:
    """
    Axis 1 -- does the latest observation itself breach a limit right now?
    Boundary-inclusivity-aware (limit_*_inclusive flags). Evaluated
    independently of standard_status -- an ignored failure is still
    classified as a real breach here.
    """
    value = latest_row["transformed_value"]

    max_inclusive = _is_inclusive(latest_row, "limit_max_inclusive")
    min_inclusive = _is_inclusive(latest_row, "limit_min_inclusive")
    upper_failure = value >= 1.0 if max_inclusive else value > 1.0
    lower_failure = value <= -1.0 if min_inclusive else value < -1.0
    if upper_failure:
        return "UPPER_FAILURE"
    if lower_failure:
        return "LOWER_FAILURE"

    max_warn_inclusive = _is_inclusive(latest_row, "limit_max_warning_inclusive")
    min_warn_inclusive = _is_inclusive(latest_row, "limit_min_warning_inclusive")
    max_warning_bound = latest_row["transformed_max_warning"]
    min_warning_bound = latest_row["transformed_min_warning"]
    upper_warning = value >= max_warning_bound if max_warn_inclusive else value > max_warning_bound
    lower_warning = value <= min_warning_bound if min_warn_inclusive else value < min_warning_bound
    if upper_warning:
        return "UPPER_WARNING"
    if lower_warning:
        return "LOWER_WARNING"

    return "NORMAL"


def _classify_drift(point_status: str, trend: Dict[str, Any], distances: Dict[str, Any], n_history: int,
                     min_history_point: int, min_history_trend: int,
                     trend_strength_min_for_escalation: float,
                     trend_progress_min_for_failure_drift: float) -> Dict[str, str]:
    """
    Combine Axis 1 (point_status) and Axis 2 (trend) into the final
    drift_status / drift_reason, respecting the min_history_point /
    min_history_trend tiers. An existing point breach always outranks a
    predicted one, and is reported even when there is too little history
    for a trend. *_FAILURE_DRIFT requires the trend to already be a
    meaningful fraction of the way toward the warning boundary with
    full-confidence history -- the severity ladder is walked, not jumped.
    """
    insufficient = n_history < min_history_point
    trend_note = (
        f" Trend not assessed: only {n_history} observation(s) available including this one "
        f"(at least {min_history_point} required)."
        if insufficient else ""
    )

    if point_status in ("UPPER_FAILURE", "LOWER_FAILURE"):
        return {
            "drift_status": point_status,
            "drift_reason": (
                f"Latest result has already breached the {point_status.replace('_', ' ').lower()} limit."
                + trend_note
            ),
        }
    if point_status in ("UPPER_WARNING", "LOWER_WARNING"):
        return {
            "drift_status": point_status,
            "drift_reason": (
                f"Latest result is currently in the {point_status.replace('_', ' ').lower()} band."
                + trend_note
            ),
        }

    if insufficient:
        return {
            "drift_status": "INSUFFICIENT_HISTORY",
            "drift_reason": (
                f"Latest result is within its warning limits, but only {n_history} observation(s) "
                f"are available including this one; at least {min_history_point} are required before "
                f"a trend can be assessed."
            ),
        }

    # point_status == NORMAL from here -- evaluate Axis 2 (predictive drift)
    direction = trend.get("direction")
    strength = trend.get("strength", np.nan)
    n_used = trend.get("n_used", 0)

    if direction in (None, "FLAT") or pd.isna(strength) or strength < trend_strength_min_for_escalation:
        return {
            "drift_status": "NORMAL",
            "drift_reason": "Latest result is within limits and no consistent trend toward a boundary is evident.",
        }

    reduced_confidence = n_history < min_history_trend
    side = "UPPER" if direction == "UP" else "LOWER"
    progress = distances.get("progress_to_warning", np.nan)
    est_warn = distances.get("est_obs_to_warning")

    escalate_to_failure = (
        not reduced_confidence
        and pd.notna(progress)
        and progress >= trend_progress_min_for_failure_drift
    )
    status = f"{side}_FAILURE_DRIFT" if escalate_to_failure else f"{side}_WARNING_DRIFT"

    reason = (
        f"Latest result is within limits, but the last {n_used} results show a consistent "
        f"{'upward' if direction == 'UP' else 'downward'} trend "
        f"(slope {trend['slope']:+.4f}/result, strength {strength:.2f}) toward the {side.lower()} "
        f"{'failure' if escalate_to_failure else 'warning'} boundary."
    )
    if pd.notna(est_warn) and not escalate_to_failure:
        reason += f" Estimated ~{est_warn:.1f} result(s) away at this rate."
    if reduced_confidence:
        reason += f" (Based on fewer than {min_history_trend} observations -- treat as indicative only.)"

    return {"drift_status": status, "drift_reason": reason}


def _is_eligible_for_history(offset: float, bound: float) -> bool:
    """
    Whether an observation's offset (transformed_value) is within +/- `bound`
    and therefore eligible to be kept in the stored history. An offset
    exactly equal to the bound is still eligible -- only a strictly greater
    magnitude is excluded.
    """
    return not (offset > bound or offset < -bound)


def _data_quality_flag(n_history: int, min_history_point: int, min_history_trend: int) -> str:
    if n_history < min_history_point:
        return f"insufficient history: <{min_history_point} points, trend not assessed"
    if n_history < min_history_trend:
        return f"reduced confidence: <{min_history_trend} history points"
    return ""


def _dedup_key(rec: Dict[str, Any]) -> tuple:
    """Within one group, the DEDUP_KEYS that vary: job, date, value."""
    return (rec[_JOB_KEY], rec["analysed_date"], rec["measured_value"])


def _dedup_sort_trim(records: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    """
    Drop duplicate observations (first occurrence wins), stably sort by
    analysed_date (same-timestamp rows keep their arrival order) and keep the
    most recent `cap`.
    """
    seen = set()
    unique = []
    for rec in records:
        key = _dedup_key(rec)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    unique.sort(key=lambda r: r["analysed_date"])
    return unique[-cap:] if len(unique) > cap else unique


def _clean_value(value):
    """Missing -> None so evaluation dicts never carry pd.NA/NaT around."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


class LCSDetector:
    """
    Detect anomalies in LCS (Laboratory Control Sample) data using a
    per-analyte rolling history.

    Method: see the module docstring. In short -- scale each result to its
    own limits (target=0, failure boundary=+/-1), process jobs in order,
    fit a robust trend against the recent history, and classify both
    "already breached" and "trending toward a boundary" into one status,
    which is then mapped to FAIL / WARNING / PASS.

    Methods:
    - detect(df, ..., history=None, persist=True) -> LCSAnomalyResult
    - build_history(df) -> pd.DataFrame  (seed a history from a batch)
    - detect_drift(df, ...) -> pd.DataFrame (legacy-shaped summary)
    """

    def __init__(self, config_path: str, debug: bool = False, history_path: Optional[str] = None):
        """
        Initialize detector with configuration.

        Args:
            config_path: Path to lcs_config.yaml
            debug: When True, emit detailed processing logs via the
                `logging` module (logger name matches this module).
            history_path: Optional override of the config's history_path
                (e.g. the POC writes to its own demo store).
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f) or {}

        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
            if not any(
                isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
                for h in logger.handlers
            ):
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
                logger.addHandler(handler)
        else:
            logger.setLevel(logging.WARNING)

        self.history_path = Path(history_path or self.config.get("history_path", "data/processed/lcs_standard_history.csv"))
        self.excluded_std_codes = set(self.config.get("excluded_std_codes", ["", "Sample", "TSV_BLANK"]))
        self.max_history_per_analyte = int(self.config.get("max_history_per_analyte", 30))
        self.min_history_point = int(self.config.get("min_history_point", 3))
        self.min_history_trend = int(self.config.get("min_history_trend", 6))
        self.trend_window = int(self.config.get("trend_window", 10))
        self.slope_eps = float(self.config.get("slope_eps", 1e-6))
        self.trend_strength_min_for_escalation = float(self.config.get("trend_strength_min_for_escalation", 0.6))
        self.trend_progress_min_for_failure_drift = float(self.config.get("trend_progress_min_for_failure_drift", 0.75))
        self.offset_exclusion_bound = float(self.config.get("offset_exclusion_bound", 2.0))

    # ── Public API ───────────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame, std_lot_code: Optional[str] = None,
               job_code: Optional[str] = None, *, history: Optional[pd.DataFrame] = None,
               persist: bool = True) -> LCSAnomalyResult:
        """
        Run historic LCS drift detection on a batch of (loader-output) rows.

        Jobs are processed chronologically; every (job, group) pair is
        evaluated once against the history known up to that job. Eligible
        observations are then added to the history. The stored history is
        only written after every group has been analysed successfully -- if
        anything fails partway through, the file on disk is left untouched.

        Args:
            df: rows in data_loader.load_qc_data()'s internal schema. Any
                rows not routed to LCS (see data_validator.select_lcs_rows)
                are ignored.
            std_lot_code: Optional std_lot_code to additionally filter to
            job_code: Optional job_code to additionally filter to
            history: Optional starting history (HISTORY_COLUMNS). When None,
                the history at `history_path` is loaded (empty if absent).
            persist: When True, the updated history is written to
                `history_path`. When False nothing is written.

        Returns:
            LCSAnomalyResult; details["results"] holds one evaluation dict
            per (job, group), in job order.
        """
        working = self._coerce(df)

        if job_code is not None and "job_code" in working.columns:
            working = working[working["job_code"] == job_code]
        if std_lot_code is not None and "std_lot_code" in working.columns:
            working = working[working["std_lot_code"] == std_lot_code]

        transformed, validation_report = self._prepare_rows(working)

        if transformed.empty:
            logger.debug("No LCS rows to analyse after routing/validation -- nothing to do.")
            return LCSAnomalyResult(
                detected=False, confidence=0.0, severity="low",
                details={"reason": "No LCS data found for the given filters",
                         "results": [], "validation_report": validation_report, "job_order": []},
                visualizable=False,
            )

        baseline = self._load_baseline(history)
        logger.debug("Starting history: %d rows", len(baseline))

        job_order = self._job_order(transformed)
        evaluations, updated_groups = self._run_job_by_job(transformed, baseline, job_order)

        final_history = _merge_updated_history(baseline, updated_groups)
        for group_key, group_hist in updated_groups.items():
            if len(group_hist) > self.max_history_per_analyte:
                raise RuntimeError(f"Group {group_key} exceeds max_history_per_analyte after trim")

        if persist:
            atomic_write_csv(final_history, self.history_path)
            logger.debug("History saved to %s: %d rows", self.history_path, len(final_history))

        return _build_result(evaluations, validation_report, job_order)

    def build_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a history store from a batch of rows WITHOUT evaluating them:
        route -> validate -> transform -> keep eligible observations ->
        de-duplicate -> keep the most recent `max_history_per_analyte` per
        group. Equivalent to the history the job-by-job run would end with,
        so it is used to seed a starting history (e.g. the POC's "older half
        of the upload").
        """
        transformed, _ = self._prepare_rows(self._coerce(df))
        if transformed.empty:
            return _ensure_history_columns(pd.DataFrame(columns=HISTORY_COLUMNS))

        parts = []
        for _, group_rows in transformed.groupby(GROUP_KEYS, sort=False, dropna=False):
            records = [r for r in group_rows.to_dict("records") if r["eligible_for_history"]]
            kept = _dedup_sort_trim(records, self.max_history_per_analyte)
            if kept:
                parts.append(pd.DataFrame(kept))
        if not parts:
            return _ensure_history_columns(pd.DataFrame(columns=HISTORY_COLUMNS))
        return _ensure_history_columns(pd.concat(parts, ignore_index=True))

    def detect_drift(self, df: pd.DataFrame, std_lot_code: Optional[str] = None,
                      job_code: Optional[str] = None) -> pd.DataFrame:
        """
        Legacy-shaped convenience wrapper around detect(), kept for callers
        built against the original job-scoped summary shape: one row per
        analyte_code from each analyte's most recent evaluation this run.
        first_warning_date / first_failure_date are not reconstructed and
        are always NaT; use `detect(...).details["results"]` for detail.
        """
        result = self.detect(df, std_lot_code=std_lot_code, job_code=job_code)
        rows = result.details.get("results", [])

        columns = ["analyte_code", "drift_detected", "drift_direction", "drift_start",
                   "severity_score", "first_warning_date", "first_failure_date", "n_observations"]
        if not rows:
            return pd.DataFrame(columns=columns)

        by_analyte: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_analyte.setdefault(r["analyte_code"], []).append(r)

        summary_rows = []
        for analyte, analyte_rows in by_analyte.items():
            latest = max(analyte_rows, key=lambda r: r["analysed_date"])
            drift_detected = latest["warning_state"] != PASS
            drift_direction = None
            if latest["drift_status"].startswith("UPPER"):
                drift_direction = "Upper"
            elif latest["drift_status"].startswith("LOWER"):
                drift_direction = "Lower"

            summary_rows.append({
                "analyte_code": analyte,
                "drift_detected": drift_detected,
                "drift_direction": drift_direction,
                "drift_start": latest["analysed_date"] if drift_detected else pd.NaT,
                "severity_score": round(abs(latest["offset"]), 4),
                "first_warning_date": pd.NaT,
                "first_failure_date": pd.NaT,
                "n_observations": latest["n_history"],
            })

        return (
            pd.DataFrame(summary_rows)
            .sort_values(["drift_detected", "severity_score"], ascending=[False, False])
            .reset_index(drop=True)
        )

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce the date/numeric columns, in case the caller passes an untyped frame."""
        working = df.copy()
        if "analysed_date" in working.columns:
            working["analysed_date"] = pd.to_datetime(working["analysed_date"], errors="coerce", format="mixed")
        for col in _NUMERIC_COLUMNS:
            if col in working.columns:
                working[col] = pd.to_numeric(working[col], errors="coerce")
        return working

    def _prepare_rows(self, working: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Route -> exclude std_codes -> validate -> transform -> eligibility flag."""
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in working.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        lcs_rows = select_lcs_rows(working)
        lcs_rows = _filter_excluded_std_codes(lcs_rows, self.excluded_std_codes)
        logger.debug("Input rows: %d; routed to LCS (after excluded std_code filter): %d",
                     len(working), len(lcs_rows))

        clean_rows, report = _validate_lcs_frame(lcs_rows)
        report = {"n_routed_to_lcs": len(lcs_rows), **report}
        logger.debug("Validation report: %s", report)

        transformed = _transform_lcs_values(clean_rows)
        for col in HISTORY_COLUMNS:
            if col not in transformed.columns:
                transformed[col] = pd.NA
        transformed[_JOB_KEY] = transformed["job_code"].fillna("").astype(str)
        transformed["eligible_for_history"] = [
            _is_eligible_for_history(float(v), self.offset_exclusion_bound)
            for v in transformed["transformed_value"]
        ]
        return transformed, report

    def _load_baseline(self, history: Optional[pd.DataFrame]) -> pd.DataFrame:
        if history is None:
            history = load_csv_or_empty(self.history_path, HISTORY_COLUMNS, parse_dates=["analysed_date"])
            logger.debug("History loaded from %s (exists=%s): %d rows",
                         self.history_path, self.history_path.is_file(), len(history))
        baseline = _ensure_history_columns(history)
        baseline["analysed_date"] = pd.to_datetime(baseline["analysed_date"], errors="coerce")
        # A stored row without a date can't be placed in time; ignore it.
        return baseline.loc[baseline["analysed_date"].notna()].reset_index(drop=True)

    @staticmethod
    def _job_order(transformed: pd.DataFrame) -> List[str]:
        """Job keys oldest-first, each job's time being its earliest analysed_date."""
        first_seen = transformed.groupby(_JOB_KEY, sort=False)["analysed_date"].min()
        return first_seen.sort_values(kind="stable").index.tolist()

    def _run_job_by_job(self, transformed: pd.DataFrame, baseline: pd.DataFrame,
                         job_order: List[str]) -> Tuple[List[Dict[str, Any]], Dict[tuple, pd.DataFrame]]:
        """
        Notebook's job-by-job simulation, computed per group: groups never
        influence each other, so walking each group through the global job
        order gives exactly the same history/evaluations as walking jobs
        across all groups -- just much faster.
        """
        job_rank = {job: i for i, job in enumerate(job_order)}

        # Plain-Python buckets (group -> job -> records, input order kept):
        # far cheaper than re-filtering a DataFrame for every job.
        baseline_groups: Dict[tuple, List[Dict[str, Any]]] = {}
        if not baseline.empty:
            base = baseline.copy()
            base[_JOB_KEY] = base["job_code"].fillna("").astype(str)
            for rec in base.to_dict("records"):
                baseline_groups.setdefault(tuple(rec[k] for k in GROUP_KEYS), []).append(rec)
            for recs in baseline_groups.values():
                recs.sort(key=lambda r: r["analysed_date"])

        new_groups: Dict[tuple, Dict[str, List[Dict[str, Any]]]] = {}
        for rec in transformed.to_dict("records"):
            key = tuple(rec[k] for k in GROUP_KEYS)
            new_groups.setdefault(key, {}).setdefault(rec[_JOB_KEY], []).append(rec)

        evaluations: List[Tuple[int, int, Dict[str, Any]]] = []
        updated_groups: Dict[tuple, pd.DataFrame] = {}

        try:
            for group_key, jobs_in_group in new_groups.items():
                stored = list(baseline_groups.get(group_key, []))
                group_jobs = sorted(jobs_in_group, key=job_rank.__getitem__)

                for job in group_jobs:
                    job_records = sorted(jobs_in_group[job], key=lambda r: r["analysed_date"])
                    evaluation = self._evaluate_job(stored, job_records)
                    evaluations.append((job_rank[job], len(evaluations), evaluation))

                    eligible = [r for r in job_records if r["eligible_for_history"]]
                    stored = _dedup_sort_trim(stored + eligible, self.max_history_per_analyte)

                updated_groups[group_key] = _ensure_history_columns(
                    pd.DataFrame(stored) if stored else pd.DataFrame(columns=HISTORY_COLUMNS)
                )
                logger.debug("Group %s: %d job(s) evaluated, final history %d row(s)",
                             group_key, len(group_jobs), len(stored))
        except Exception:
            logger.exception("LCS drift detection failed before completing analysis -- history left untouched.")
            raise

        evaluations.sort(key=lambda t: (t[0], t[1]))
        return [e for _, _, e in evaluations], updated_groups

    def _evaluate_job(self, stored: List[Dict[str, Any]], job_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate one job's observations for one group against the history
        known before it (point-in-time guarded), returning one evaluation
        dict for the job's latest observation.
        """
        cutoff = job_records[-1]["analysed_date"]
        prior = [r for r in stored if r["analysed_date"] <= cutoff]
        candidate = _dedup_sort_trim(prior + job_records, self.max_history_per_analyte)

        latest = candidate[-1]
        offsets = [float(r["transformed_value"]) for r in candidate]
        n_history = len(candidate)

        trend = _calculate_trend(offsets, window=self.trend_window, slope_eps=self.slope_eps)
        distances = _calculate_threshold_distance(latest, trend, self.trend_strength_min_for_escalation)
        point_status = _classify_point_status(latest)
        drift = _classify_drift(
            point_status, trend, distances, n_history,
            self.min_history_point, self.min_history_trend,
            self.trend_strength_min_for_escalation, self.trend_progress_min_for_failure_drift,
        )

        offset = float(latest["transformed_value"])
        trend_start = n_history - trend["n_used"]
        history_window = [
            {
                "sequence": i,
                "analysed_date": r["analysed_date"],
                "offset": float(r["transformed_value"]),
                "measured_value": float(r["measured_value"]),
                "job_code": _clean_value(r.get("job_code")),
                "in_trend_window": i >= trend_start,
                "is_current": i == n_history - 1,
            }
            for i, r in enumerate(candidate)
        ]

        evaluation = {
            # identity
            "job_code": _clean_value(latest.get("job_code")),
            "instrument_id": _clean_value(latest.get("instrument_id")),
            "analyte_code": latest["analyte_code"],
            "std_code": latest["std_code"],
            "std_lot_code": _clean_value(latest.get("std_lot_code")),
            "scheme_code": latest["scheme_code"],
            "unit_code": _clean_value(latest.get("unit_code")),
            # current observation
            "analysed_date": latest["analysed_date"],
            "measured_value": float(latest["measured_value"]),
            "target_value": float(latest["target_value"]),
            "limit_min": float(latest["limit_min"]),
            "limit_max": float(latest["limit_max"]),
            "limit_min_warning": float(latest["limit_min_warning"]),
            "limit_max_warning": float(latest["limit_max_warning"]),
            "standard_status": _clean_value(latest.get("standard_status")),
            "offset": offset,
            "transformed_max_warning": float(latest["transformed_max_warning"]),
            "transformed_min_warning": float(latest["transformed_min_warning"]),
            "eligible_for_history": _is_eligible_for_history(offset, self.offset_exclusion_bound),
            # history / trend
            "n_history": n_history,
            "oldest_date": candidate[0]["analysed_date"],
            "point_status": point_status,
            "trend_direction": trend["direction"],
            "trend_slope": trend["slope"],
            "trend_intercept": trend["intercept"],
            "trend_strength": trend["strength"],
            "trend_n_used": trend["n_used"],
            "trend_ci_low": trend["ci_low"],
            "trend_ci_high": trend["ci_high"],
            "distance_to_warning": distances["distance_to_warning"],
            "distance_to_failure": distances["distance_to_failure"],
            "progress_to_warning": distances["progress_to_warning"],
            "est_obs_to_warning": distances["est_obs_to_warning"],
            "est_obs_to_failure": distances["est_obs_to_failure"],
            # outcome
            "drift_status": drift["drift_status"],
            "drift_reason": drift["drift_reason"],
            "warning_state": lcs_warning_state(drift["drift_status"]),
            "data_quality_flag": _data_quality_flag(n_history, self.min_history_point, self.min_history_trend),
            "history_window": history_window,
        }

        logger.debug(
            "  %s/%s/%s job %s @ %s: offset=%.4f -> %s (%s)%s",
            evaluation["analyte_code"], evaluation["std_code"], evaluation["scheme_code"],
            evaluation["job_code"], evaluation["analysed_date"], offset,
            evaluation["drift_status"], evaluation["warning_state"],
            "" if evaluation["eligible_for_history"] else " [excluded from history: |offset| > bound]",
        )
        return evaluation


def _merge_updated_history(history_df: pd.DataFrame, updated_group_histories: Dict[tuple, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine untouched groups (passed through unchanged) with the freshly
    updated touched groups into one schema-conformant history frame, ready
    to persist. Never drops or reorders rows belonging to a group that
    wasn't touched this run.
    """
    touched_mask = pd.Series(False, index=history_df.index)
    for group_key in updated_group_histories:
        group_mask = pd.Series(True, index=history_df.index)
        for col, val in zip(GROUP_KEYS, group_key):
            group_mask &= (history_df[col] == val)
        touched_mask |= group_mask

    untouched_history = history_df.loc[~touched_mask]
    parts = [p for p in [untouched_history, *updated_group_histories.values()] if not p.empty]
    combined = pd.concat(parts, ignore_index=True) if parts else history_df.iloc[0:0]
    return _ensure_history_columns(combined)


def _build_result(evaluations: List[Dict[str, Any]], validation_report: Dict[str, int],
                  job_order: List[str]) -> LCSAnomalyResult:
    """Aggregate this run's evaluations into one LCSAnomalyResult."""
    if not evaluations:
        return LCSAnomalyResult(
            detected=False, confidence=0.0, severity="low",
            details={"reason": "No analysable LCS observations after validation",
                     "results": [], "validation_report": validation_report, "job_order": job_order},
            visualizable=False,
        )

    flagged = [e for e in evaluations if e["warning_state"] != PASS]
    most_severe = min(
        evaluations, key=lambda e: (STATE_RANK.get(e["warning_state"], 9), -abs(e["offset"]))
    )
    state = most_severe["warning_state"]
    detected = state != PASS

    n_analysed = len(evaluations)
    reason = (
        f"{len(flagged)} of {n_analysed} analysed job/analyte evaluation(s) flagged"
        if detected else
        f"No drift/anomalies detected in {n_analysed} analysed job/analyte evaluation(s)"
    )
    logger.debug("Final result counts: %d flagged, %d pass (of %d analysed)",
                 len(flagged), n_analysed - len(flagged), n_analysed)

    details = {
        "reason": reason,
        "results": evaluations,
        "most_severe": most_severe if detected else None,
        "validation_report": validation_report,
        "job_order": job_order,
    }

    return LCSAnomalyResult(detected=detected, confidence=_STATE_CONFIDENCE[state],
                             severity=_STATE_TO_RESULT_SEVERITY[state],
                             details=details, visualizable=True)
