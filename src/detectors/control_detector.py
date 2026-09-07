"""
LCS (Laboratory Control Sample) Historic Drift Detector

Detects deviations in control samples that verify method accuracy and
stability, tracked **across jobs over time** via a persistent, per-analyte
rolling history -- not just within a single analytical run.

This supersedes the job-scoped approach this module used to implement
(originally from notebooks/LCS_drift_detection.ipynb): that method
normalised results against warning bounds and rolling-averaged them within
one JOB_CODE, so it could never see a slow drift that plays out across weeks
of separate jobs, since every job started a fresh rolling window.

Method (ported from notebooks/LCS_drift_detection_historic.ipynb, see
that notebook and its design doc for the full rationale):
- Independent history per (ANALYTE_CODE, STD_CODE, SCHEME_CODE) group,
  capped at `max_history_per_analyte` (default 30) observations, persisted
  to a CSV between calls.
- Each observation is scaled per-row against its own limits so
  INTERNAL_TARGET_VALUE -> 0, INTERNAL_MAX_VALUE -> +1, INTERNAL_MIN_VALUE
  -> -1 (piecewise: divide by the upper span above target, the lower span
  below it). This scaled value is what the rest of this module calls the
  "offset" (TRANSFORMED_VALUE).
- A Theil-Sen slope (robust to a single outlier, unlike OLS) over the most
  recent `trend_window` observations' equal-step observation order (not
  elapsed calendar time -- see the historic notebook's rationale) gives a
  trend direction/strength.
- Two axes are combined into one DRIFT_STATUS/DRIFT_SEVERITY: has the
  latest result already breached a limit (classify_point_status), and/or
  is a consistent trend heading toward one (classify_drift).

Offset-exclusion rule (new in this port, not present in either notebook):
an observation is always analysed and flagged normally, but is only folded
into the persisted history if its offset falls within
+/- `offset_exclusion_bound` (default 2.0 -- twice the target-to-failure
span). A wilder reading is treated as analysed-but-not-representative, so a
one-off bad result can't distort future trend fits. An offset of exactly
+/- the bound is still eligible (only a strictly greater magnitude is
excluded).

Configuration: config/lcs_config.yaml
- history_path: where the persisted rolling history CSV lives
- excluded_std_codes: STD_CODE values that aren't genuine reference materials
- max_history_per_analyte: rolling-history cap per group (default 30)
- min_history_point / min_history_trend: two-tier minimum-data thresholds
- trend_window, slope_eps, trend_strength_min_for_escalation,
  trend_progress_min_for_failure_drift: trend/classification tuning
- offset_exclusion_bound: the new history-eligibility cutoff
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import theilslopes

from .base_detector import LCSAnomalyResult
from ..utils import atomic_write_csv, load_csv_or_empty

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ── Grouping ─────────────────────────────────────────────────────────────────
# One independent historic time series = one (ANALYTE_CODE, STD_CODE,
# SCHEME_CODE) combination -- see notebooks/LCS_drift_detection_historic.ipynb
# for why STD_CODE (not STD_LOT_CODE) and SCHEME_CODE are the correct identity
# fields, and why JOB_CODE is deliberately excluded from the grouping key.
GROUP_KEYS = ["ANALYTE_CODE", "STD_CODE", "SCHEME_CODE"]

# Columns _validate_lcs_frame() requires to be present before any row-level
# filtering; rows with nulls in the limit/date columns are dropped, not the
# whole run failing -- see _validate_lcs_frame().
REQUIRED_COLUMNS = [
    "ANALYTICAL_TYPE", "STD_CODE", "SCHEME_CODE", "ANALYTE_CODE",
    "ANALYSED_DATE", "NUMERIC_FINAL_VALUE",
    "INTERNAL_TARGET_VALUE", "INTERNAL_MAX_VALUE", "INTERNAL_MIN_VALUE",
    "INTERNAL_MAX_WARNING_VALUE", "INTERNAL_MIN_WARNING_VALUE",
]

# Numeric/date columns coerced to their proper dtype on ingestion, in case the
# caller passes a raw, untyped DataFrame (e.g. straight off pd.read_csv).
_NUMERIC_COLUMNS = [
    "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE", "INTERNAL_MAX_VALUE", "INTERNAL_MIN_VALUE",
    "INTERNAL_MAX_WARNING_VALUE", "INTERNAL_MIN_WARNING_VALUE",
]

# Columns persisted in the history store -- raw traceability columns plus the
# transformed value/limits computed at ingestion time (both are kept: raw for
# display/audit, transformed to avoid recomputing against possibly-since-
# changed limits).
HISTORY_COLUMNS = [
    "ANALYTE_CODE", "STD_CODE", "STD_LOT_CODE", "SCHEME_CODE", "JOB_CODE",
    "ANALYSED_DATE", "UNIT_CODE", "STANDARD_STATUS",
    "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE", "INTERNAL_MAX_VALUE", "INTERNAL_MIN_VALUE",
    "INTERNAL_MAX_WARNING_VALUE", "INTERNAL_MIN_WARNING_VALUE",
    "INTERNAL_MAX_INCLUSIVE", "INTERNAL_MIN_INCLUSIVE",
    "INTERNAL_MAX_WARNING_INCLUSIVE", "INTERNAL_MIN_WARNING_INCLUSIVE",
    "TRANSFORMED_VALUE", "TRANSFORMED_TARGET", "TRANSFORMED_MAX", "TRANSFORMED_MIN",
    "TRANSFORMED_MAX_WARNING", "TRANSFORMED_MIN_WARNING",
]

_SEVERITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "NONE": 3}
_SEVERITY_TO_RESULT = {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "NONE": "low"}
_SEVERITY_CONFIDENCE = {"CRITICAL": 100.0, "HIGH": 85.0, "MEDIUM": 65.0, "NONE": 95.0}


def _ensure_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return df restricted/reordered to HISTORY_COLUMNS, adding any missing ones as NA."""
    df = df.copy()
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[HISTORY_COLUMNS]


def _filter_standard_rows(df: pd.DataFrame, excluded_std_codes) -> pd.DataFrame:
    """
    Restrict to ANALYTICAL_TYPE == "Standard" rows with a usable STD_CODE.

    Excludes STD_CODE values that are not genuine reference-material
    identifiers (blank, "Sample", "TSV_BLANK" by default).
    """
    std_code = df["STD_CODE"].fillna("").astype(str).str.strip()
    mask = (df["ANALYTICAL_TYPE"] == "Standard") & (~std_code.isin(excluded_std_codes))
    return df.loc[mask].copy()


def _validate_lcs_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validate filtered Standard rows ahead of transformation.

    Checks REQUIRED_COLUMNS are present (raises if not), then drops rows with
    a missing ANALYSED_DATE, a missing target or any limit value, a
    zero-span limit (target == max or target == min -- would divide by
    zero), or a target outside [INTERNAL_MIN_VALUE, INTERNAL_MAX_VALUE]
    (data-entry error).

    Returns (clean_df, report) where `report` counts what was dropped and
    why, so data-quality issues stay visible rather than silently vanishing.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    report = {"n_input": len(df)}
    clean = df.copy()

    missing_date_mask = clean["ANALYSED_DATE"].isna()
    report["n_dropped_missing_date"] = int(missing_date_mask.sum())
    clean = clean.loc[~missing_date_mask]

    limit_cols = ["INTERNAL_TARGET_VALUE", "INTERNAL_MAX_VALUE", "INTERNAL_MIN_VALUE",
                  "INTERNAL_MAX_WARNING_VALUE", "INTERNAL_MIN_WARNING_VALUE"]
    missing_limits_mask = clean[limit_cols].isna().any(axis=1)
    report["n_dropped_missing_limits"] = int(missing_limits_mask.sum())
    clean = clean.loc[~missing_limits_mask]

    zero_span_mask = (
        (clean["INTERNAL_MAX_VALUE"] == clean["INTERNAL_TARGET_VALUE"]) |
        (clean["INTERNAL_TARGET_VALUE"] == clean["INTERNAL_MIN_VALUE"])
    )
    report["n_dropped_zero_span"] = int(zero_span_mask.sum())
    clean = clean.loc[~zero_span_mask]

    target_outside_mask = (
        (clean["INTERNAL_TARGET_VALUE"] > clean["INTERNAL_MAX_VALUE"]) |
        (clean["INTERNAL_TARGET_VALUE"] < clean["INTERNAL_MIN_VALUE"])
    )
    report["n_dropped_target_outside_range"] = int(target_outside_mask.sum())
    clean = clean.loc[~target_outside_mask].copy()

    report["n_output"] = len(clean)
    return clean, report


def _transform_lcs_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the piecewise LCS transform: INTERNAL_TARGET_VALUE -> 0,
    INTERNAL_MAX_VALUE -> +1, INTERNAL_MIN_VALUE -> -1, computed per row
    using that row's OWN target/limits. A value on or above target is
    scaled by the upper span (max - target); a value below target is scaled
    by the lower span (target - min). TRANSFORMED_VALUE is what the rest of
    this module calls "offset".

    Assumes `df` has already passed _validate_lcs_frame() (no missing or
    zero-span limits).
    """
    out = df.copy()
    target = out["INTERNAL_TARGET_VALUE"]
    upper_span = out["INTERNAL_MAX_VALUE"] - target
    lower_span = target - out["INTERNAL_MIN_VALUE"]

    value = out["NUMERIC_FINAL_VALUE"]
    span_for_value = np.where(value >= target, upper_span, lower_span)
    out["TRANSFORMED_VALUE"] = (value - target) / span_for_value

    out["TRANSFORMED_TARGET"] = 0.0
    out["TRANSFORMED_MAX"] = 1.0
    out["TRANSFORMED_MIN"] = -1.0
    out["TRANSFORMED_MAX_WARNING"] = (out["INTERNAL_MAX_WARNING_VALUE"] - target) / upper_span
    out["TRANSFORMED_MIN_WARNING"] = (out["INTERNAL_MIN_WARNING_VALUE"] - target) / lower_span

    return out


def _assign_sequence_position(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stably sort one group's rows by ANALYSED_DATE and assign equal-step
    ordinal positions 0..n-1. Same-timestamp rows land on consecutive
    positions (stable sort) rather than overlapping -- the shared x-axis
    basis for the trend fit.
    """
    ordered = group_df.sort_values("ANALYSED_DATE", kind="stable").reset_index(drop=True)
    ordered["SEQUENCE_INDEX"] = np.arange(len(ordered))
    return ordered


def _calculate_trend(group_df: pd.DataFrame, window: int, slope_eps: float) -> Dict[str, Any]:
    """
    Theil-Sen slope of TRANSFORMED_VALUE over the most recent `window`
    observations of one group, x = equal-step observation order (not
    elapsed time). Theil-Sen (median of pairwise slopes) is used instead of
    OLS because it is resistant to a single isolated outlier tilting the fit.
    """
    ordered = _assign_sequence_position(group_df)
    recent = ordered.tail(window)
    n = len(recent)

    if n < 2:
        return {"slope": np.nan, "direction": "FLAT", "strength": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "n_used": n}

    x = recent["SEQUENCE_INDEX"].to_numpy(dtype=float)
    y = recent["TRANSFORMED_VALUE"].to_numpy(dtype=float)

    slope, _intercept, lo, hi = theilslopes(y, x)

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

    return {"slope": float(slope), "direction": direction, "strength": strength,
            "ci_low": float(lo), "ci_high": float(hi), "n_used": n}


def _calculate_threshold_distance(latest_row: pd.Series, trend: Dict[str, Any],
                                   trend_strength_min_for_escalation: float) -> Dict[str, Any]:
    """
    Signed-magnitude distance from the latest transformed value to the
    relevant warning/failure boundary given the trend direction, plus an
    advisory estimated-observations-to-boundary projection when the trend
    is consistent enough to trust.
    """
    value = latest_row["TRANSFORMED_VALUE"]
    direction = trend.get("direction")

    if direction == "UP":
        warning_bound, failure_bound = latest_row["TRANSFORMED_MAX_WARNING"], latest_row["TRANSFORMED_MAX"]
    elif direction == "DOWN":
        warning_bound, failure_bound = latest_row["TRANSFORMED_MIN_WARNING"], latest_row["TRANSFORMED_MIN"]
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


def _is_inclusive(row: pd.Series, col: str) -> bool:
    """Missing inclusivity flags default to inclusive."""
    val = row.get(col)
    if pd.isna(val):
        return True
    return str(val).strip().upper() == "Y"


def _classify_point_status(latest_row: pd.Series) -> str:
    """
    Axis 1 -- does the latest observation itself breach a limit right now?
    Boundary-inclusivity-aware (INTERNAL_*_INCLUSIVE flags). Evaluated
    independently of STANDARD_STATUS -- an ignored failure is still
    classified as a real breach here.
    """
    value = latest_row["TRANSFORMED_VALUE"]

    max_inclusive = _is_inclusive(latest_row, "INTERNAL_MAX_INCLUSIVE")
    min_inclusive = _is_inclusive(latest_row, "INTERNAL_MIN_INCLUSIVE")
    upper_failure = value >= 1.0 if max_inclusive else value > 1.0
    lower_failure = value <= -1.0 if min_inclusive else value < -1.0
    if upper_failure:
        return "UPPER_FAILURE"
    if lower_failure:
        return "LOWER_FAILURE"

    max_warn_inclusive = _is_inclusive(latest_row, "INTERNAL_MAX_WARNING_INCLUSIVE")
    min_warn_inclusive = _is_inclusive(latest_row, "INTERNAL_MIN_WARNING_INCLUSIVE")
    max_warning_bound = latest_row["TRANSFORMED_MAX_WARNING"]
    min_warning_bound = latest_row["TRANSFORMED_MIN_WARNING"]
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
    DRIFT_STATUS / DRIFT_SEVERITY / DRIFT_REASON, respecting the
    min_history_point / min_history_trend tiers. An existing point breach
    always outranks a predicted one. *_FAILURE_DRIFT requires the trend to
    already be a meaningful fraction of the way toward the warning boundary
    with full-confidence history -- the severity ladder is walked, not
    jumped.
    """
    if n_history < min_history_point:
        return {
            "drift_status": "INSUFFICIENT_HISTORY",
            "drift_severity": "NONE",
            "drift_reason": (
                f"Only {n_history} historical observation(s) available; at least "
                f"{min_history_point} are required before any status can be assessed."
            ),
        }

    if point_status in ("UPPER_FAILURE", "LOWER_FAILURE"):
        return {
            "drift_status": point_status,
            "drift_severity": "CRITICAL",
            "drift_reason": f"Latest result has already breached the {point_status.replace('_', ' ').lower()} limit.",
        }
    if point_status in ("UPPER_WARNING", "LOWER_WARNING"):
        return {
            "drift_status": point_status,
            "drift_severity": "MEDIUM",
            "drift_reason": f"Latest result is currently in the {point_status.replace('_', ' ').lower()} band.",
        }

    # point_status == NORMAL from here -- evaluate Axis 2 (predictive drift)
    direction = trend.get("direction")
    strength = trend.get("strength", np.nan)
    n_used = trend.get("n_used", 0)

    if direction in (None, "FLAT") or pd.isna(strength) or strength < trend_strength_min_for_escalation:
        return {
            "drift_status": "NORMAL",
            "drift_severity": "NONE",
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
    severity = "HIGH" if escalate_to_failure else "MEDIUM"

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

    return {"drift_status": status, "drift_severity": severity, "drift_reason": reason}


def _is_eligible_for_history(offset: float, bound: float) -> bool:
    """
    Whether an observation's offset (TRANSFORMED_VALUE) is within
    +/- `bound` and therefore eligible to be folded into persisted history.
    An offset exactly equal to the bound is still eligible -- only a
    strictly greater magnitude is excluded.
    """
    return not (offset > bound or offset < -bound)


def _update_group_history(existing_history: pd.DataFrame, new_rows: pd.DataFrame, *,
                           max_per_group: int, trend_window: int, slope_eps: float,
                           min_history_point: int, min_history_trend: int,
                           trend_strength_min_for_escalation: float,
                           trend_progress_min_for_failure_drift: float,
                           offset_bound: float) -> Tuple[pd.DataFrame, List[Dict[str, Any]], int]:
    """
    Analyse every new observation for one (ANALYTE_CODE, STD_CODE,
    SCHEME_CODE) group against the SAME baseline history (`existing_history`,
    as it stood before this call) -- so multiple new observations for one
    analyte in a single run never see each other's results, only what was
    already known before the run. Eligible observations
    (`_is_eligible_for_history`) are then folded into the baseline together
    and trimmed to `max_per_group`, oldest-first.

    Returns (updated_history_df, per_observation_results, n_oldest_removed).
    `per_observation_results` includes ineligible observations too -- they
    were still analysed, just not retained.
    """
    baseline = existing_history.sort_values("ANALYSED_DATE", kind="stable").reset_index(drop=True)
    new_rows_sorted = new_rows.sort_values("ANALYSED_DATE", kind="stable").reset_index(drop=True)

    results: List[Dict[str, Any]] = []
    eligible_row_frames: List[pd.DataFrame] = []

    for i in range(len(new_rows_sorted)):
        row_df = new_rows_sorted.iloc[[i]]  # single-row DataFrame; preserves column dtypes
        candidate = pd.concat([baseline, row_df], ignore_index=True)
        candidate = candidate.sort_values("ANALYSED_DATE", kind="stable").reset_index(drop=True)

        n_history = len(candidate)
        trend = _calculate_trend(candidate, window=trend_window, slope_eps=slope_eps)
        latest = candidate.iloc[-1]

        distances = _calculate_threshold_distance(latest, trend, trend_strength_min_for_escalation)
        point_status = _classify_point_status(latest)
        drift = _classify_drift(
            point_status, trend, distances, n_history,
            min_history_point, min_history_trend,
            trend_strength_min_for_escalation, trend_progress_min_for_failure_drift,
        )

        offset = float(latest["TRANSFORMED_VALUE"])
        eligible = _is_eligible_for_history(offset, offset_bound)

        results.append({
            "ANALYTE_CODE": latest["ANALYTE_CODE"],
            "STD_CODE": latest["STD_CODE"],
            "SCHEME_CODE": latest["SCHEME_CODE"],
            "STD_LOT_CODE": latest.get("STD_LOT_CODE"),
            "JOB_CODE": latest.get("JOB_CODE"),
            "ANALYSED_DATE": latest["ANALYSED_DATE"],
            "NUMERIC_FINAL_VALUE": latest["NUMERIC_FINAL_VALUE"],
            "OFFSET": offset,
            "N_HISTORY": n_history,
            "POINT_STATUS": point_status,
            "TREND_DIRECTION": trend["direction"],
            "TREND_SLOPE": trend["slope"],
            "TREND_STRENGTH": trend["strength"],
            "TREND_N_USED": trend["n_used"],
            "DISTANCE_TO_WARNING": distances["distance_to_warning"],
            "DISTANCE_TO_FAILURE": distances["distance_to_failure"],
            "DRIFT_STATUS": drift["drift_status"],
            "DRIFT_SEVERITY": drift["drift_severity"],
            "DRIFT_REASON": drift["drift_reason"],
            "ELIGIBLE_FOR_HISTORY": eligible,
        })

        logger.debug(
            "  observation %s/%s/%s @ %s: offset=%.4f -> %s (%s)%s",
            latest["ANALYTE_CODE"], latest["STD_CODE"], latest["SCHEME_CODE"], latest["ANALYSED_DATE"],
            offset, drift["drift_status"], drift["drift_severity"],
            "" if eligible else " [excluded from history: |offset| > bound]",
        )

        if eligible:
            eligible_row_frames.append(row_df)

    if eligible_row_frames:
        additions = pd.concat(eligible_row_frames, ignore_index=True)
        updated = pd.concat([baseline, additions], ignore_index=True)
        updated = updated.sort_values("ANALYSED_DATE", kind="stable").reset_index(drop=True)
    else:
        updated = baseline

    oldest_removed = 0
    if len(updated) > max_per_group:
        oldest_removed = len(updated) - max_per_group
        updated = updated.iloc[oldest_removed:].reset_index(drop=True)

    return updated, results, oldest_removed


class LCSDetector:
    """
    Detect anomalies in LCS (Laboratory Control Sample) data using a
    persistent, per-analyte rolling history.

    Method: see the module docstring. In short -- scale each result to its
    own limits (target=0, failure boundary=+/-1), fit a robust trend against
    a rolling window of recent history, and classify both "already breached"
    and "trending toward a boundary" into one status. History is capped at
    `max_history_per_analyte` observations per (ANALYTE_CODE, STD_CODE,
    SCHEME_CODE) group and persisted to a CSV between calls; an observation
    with an extreme offset is analysed and flagged but excluded from that
    persisted history.

    Methods:
    - detect(df, std_lot_code=None, job_code=None) -> LCSAnomalyResult
    - detect_drift(df, std_lot_code=None, job_code=None) -> pd.DataFrame
      (legacy-shaped summary, kept for interface compatibility)
    """

    def __init__(self, config_path: str, debug: bool = False):
        """
        Initialize detector with configuration.

        Args:
            config_path: Path to lcs_config.yaml
            debug: When True, emit detailed processing logs via the
                `logging` module (logger name matches this module). When
                False (default), this detector produces no log/console
                output beyond what the caller's own logging config allows.
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

        self.history_path = Path(self.config.get("history_path", "data/processed/lcs_standard_history.csv"))
        self.excluded_std_codes = set(self.config.get("excluded_std_codes", ["", "Sample", "TSV_BLANK"]))
        self.max_history_per_analyte = int(self.config.get("max_history_per_analyte", 30))
        self.min_history_point = int(self.config.get("min_history_point", 3))
        self.min_history_trend = int(self.config.get("min_history_trend", 6))
        self.trend_window = int(self.config.get("trend_window", 10))
        self.slope_eps = float(self.config.get("slope_eps", 1e-6))
        self.trend_strength_min_for_escalation = float(self.config.get("trend_strength_min_for_escalation", 0.6))
        self.trend_progress_min_for_failure_drift = float(self.config.get("trend_progress_min_for_failure_drift", 0.75))
        self.offset_exclusion_bound = float(self.config.get("offset_exclusion_bound", 2.0))

    def detect(self, df: pd.DataFrame, std_lot_code: Optional[str] = None,
               job_code: Optional[str] = None) -> LCSAnomalyResult:
        """
        Run historic LCS drift detection for the given data, optionally
        restricted to one job/standard-lot.

        Every analyte observation in scope is analysed against its group's
        persisted history (loaded once at the start of this call) and
        flagged accordingly; eligible observations are then folded into
        that history and written back -- but only after every group has
        been analysed successfully. If anything fails partway through, the
        on-disk history file is left untouched.

        Args:
            df: DataFrame with Standard records (any dtype hygiene -- key
                numeric/date columns are coerced on ingestion)
            std_lot_code: Optional STD_LOT_CODE to additionally filter to
            job_code: Optional JOB_CODE to additionally filter to

        Returns:
            LCSAnomalyResult with drift status, severity, and per-observation
            details.
        """
        logger.debug("Historic CSV location: %s", self.history_path)

        working = df.copy()
        if "ANALYSED_DATE" in working.columns:
            working["ANALYSED_DATE"] = pd.to_datetime(working["ANALYSED_DATE"], errors="coerce")
        for col in _NUMERIC_COLUMNS:
            if col in working.columns:
                working[col] = pd.to_numeric(working[col], errors="coerce")

        if job_code is not None:
            working = working[working["JOB_CODE"] == job_code]
        if std_lot_code is not None:
            working = working[working["STD_LOT_CODE"] == std_lot_code]

        standard_rows = _filter_standard_rows(working, self.excluded_std_codes)
        logger.debug("New input records: %d; after Standard/excluded-STD_CODE filter: %d",
                     len(working), len(standard_rows))

        if standard_rows.empty:
            logger.debug("No Standard rows to analyse after filtering -- nothing to do.")
            return LCSAnomalyResult(
                detected=False, confidence=0.0, severity="low",
                details={"reason": "No LCS data found for the given filters"},
                visualizable=False,
            )

        clean_rows, validation_report = _validate_lcs_frame(standard_rows)
        logger.debug("Validation report: %s", validation_report)

        transformed_new = _transform_lcs_values(clean_rows)

        history_df = load_csv_or_empty(self.history_path, HISTORY_COLUMNS, parse_dates=["ANALYSED_DATE"])
        logger.debug("Historic CSV loaded from %s (exists=%s): %d total rows",
                     self.history_path, self.history_path.is_file(), len(history_df))

        touched_groups = list(transformed_new[GROUP_KEYS].drop_duplicates().itertuples(index=False, name=None))
        logger.debug("Analytes/groups being processed this run: %s", touched_groups)

        all_results: List[Dict[str, Any]] = []
        updated_group_histories: Dict[tuple, pd.DataFrame] = {}
        n_excluded = 0
        n_accepted = 0

        try:
            for group_key in touched_groups:
                existing_group_history = _select_group(history_df, group_key)
                new_group_rows = _select_group(transformed_new, group_key)

                logger.debug("Group %s: %d historic obs before this run, %d new observation(s)",
                             group_key, len(existing_group_history), len(new_group_rows))

                updated_history, group_results, oldest_removed = _update_group_history(
                    existing_group_history, new_group_rows,
                    max_per_group=self.max_history_per_analyte,
                    trend_window=self.trend_window,
                    slope_eps=self.slope_eps,
                    min_history_point=self.min_history_point,
                    min_history_trend=self.min_history_trend,
                    trend_strength_min_for_escalation=self.trend_strength_min_for_escalation,
                    trend_progress_min_for_failure_drift=self.trend_progress_min_for_failure_drift,
                    offset_bound=self.offset_exclusion_bound,
                )

                n_accepted += sum(1 for r in group_results if r["ELIGIBLE_FOR_HISTORY"])
                n_excluded += sum(1 for r in group_results if not r["ELIGIBLE_FOR_HISTORY"])

                if oldest_removed:
                    logger.debug("Group %s: evicted %d oldest record(s) to stay within the %d-observation cap",
                                 group_key, oldest_removed, self.max_history_per_analyte)
                logger.debug("Group %s: final historic record count = %d", group_key, len(updated_history))

                updated_group_histories[group_key] = updated_history
                all_results.extend(group_results)
        except Exception:
            logger.exception("LCS historic drift detection failed before completing analysis -- "
                              "historic CSV left untouched.")
            raise

        final_history = _merge_updated_history(history_df, updated_group_histories, touched_groups)

        for group_key, group_hist in updated_group_histories.items():
            if len(group_hist) > self.max_history_per_analyte:
                raise RuntimeError(f"Group {group_key} exceeds max_history_per_analyte after trim")

        atomic_write_csv(final_history, self.history_path)
        logger.debug("Historic CSV saved to %s: %d total rows across %d groups",
                     self.history_path, len(final_history), final_history[GROUP_KEYS].drop_duplicates().shape[0])
        logger.debug("Records excluded from history this run (|offset| > %.2f): %d; accepted: %d",
                     self.offset_exclusion_bound, n_excluded, n_accepted)

        return _build_result(all_results, validation_report)

    def detect_drift(self, df: pd.DataFrame, std_lot_code: Optional[str] = None,
                      job_code: Optional[str] = None) -> pd.DataFrame:
        """
        Legacy-shaped convenience wrapper around detect(), kept for callers
        built against the original job-scoped summary shape. Runs the same
        historic pipeline, then folds each analyte's most recent
        observation this run into one row per ANALYTE_CODE, mapped from the
        new DRIFT_STATUS/DRIFT_SEVERITY fields.

        Note: unlike the original detect_drift(), first_warning_date /
        first_failure_date are not reconstructed here (they required
        STANDARD_STATUS history this method doesn't retain) and are always
        NaT; use `detect(...).details["results"]` for the full per-
        observation breakdown.
        """
        result = self.detect(df, std_lot_code=std_lot_code, job_code=job_code)
        rows = result.details.get("results", [])

        columns = ["ANALYTE_CODE", "drift_detected", "drift_direction", "drift_start",
                   "severity_score", "first_warning_date", "first_failure_date", "n_observations"]
        if not rows:
            return pd.DataFrame(columns=columns)

        by_analyte: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_analyte.setdefault(r["ANALYTE_CODE"], []).append(r)

        summary_rows = []
        for analyte, analyte_rows in by_analyte.items():
            latest = max(analyte_rows, key=lambda r: r["ANALYSED_DATE"])
            drift_detected = latest["DRIFT_SEVERITY"] in ("CRITICAL", "HIGH", "MEDIUM")
            drift_direction = None
            if latest["DRIFT_STATUS"].startswith("UPPER"):
                drift_direction = "Upper"
            elif latest["DRIFT_STATUS"].startswith("LOWER"):
                drift_direction = "Lower"

            summary_rows.append({
                "ANALYTE_CODE": analyte,
                "drift_detected": drift_detected,
                "drift_direction": drift_direction,
                "drift_start": latest["ANALYSED_DATE"] if drift_detected else pd.NaT,
                "severity_score": round(abs(latest["OFFSET"]), 4),
                "first_warning_date": pd.NaT,
                "first_failure_date": pd.NaT,
                "n_observations": latest["N_HISTORY"],
            })

        return (
            pd.DataFrame(summary_rows)
            .sort_values(["drift_detected", "severity_score"], ascending=[False, False])
            .reset_index(drop=True)
        )


def _select_group(df: pd.DataFrame, group_key: tuple) -> pd.DataFrame:
    """Rows of `df` matching one (ANALYTE_CODE, STD_CODE, SCHEME_CODE) group_key."""
    mask = pd.Series(True, index=df.index)
    for col, val in zip(GROUP_KEYS, group_key):
        mask &= (df[col] == val)
    return df.loc[mask]


def _merge_updated_history(history_df: pd.DataFrame, updated_group_histories: Dict[tuple, pd.DataFrame],
                            touched_groups: list) -> pd.DataFrame:
    """
    Combine untouched groups (passed through unchanged) with the freshly
    updated touched groups into one schema-conformant history frame, ready
    to persist. Never drops or reorders rows belonging to a group that
    wasn't touched this run.
    """
    touched_mask = pd.Series(False, index=history_df.index)
    for group_key in touched_groups:
        group_mask = pd.Series(True, index=history_df.index)
        for col, val in zip(GROUP_KEYS, group_key):
            group_mask &= (history_df[col] == val)
        touched_mask |= group_mask

    untouched_history = history_df.loc[~touched_mask]
    parts = [untouched_history] + list(updated_group_histories.values())
    combined = pd.concat(parts, ignore_index=True) if parts else history_df.iloc[0:0]
    return _ensure_history_columns(combined)


def _build_result(all_results: List[Dict[str, Any]], validation_report: Dict[str, int]) -> LCSAnomalyResult:
    """Aggregate per-observation results from this run into one LCSAnomalyResult."""
    if not all_results:
        return LCSAnomalyResult(
            detected=False, confidence=0.0, severity="low",
            details={"reason": "No analysable LCS observations after validation"},
            visualizable=False,
        )

    flagged = [r for r in all_results if r["DRIFT_SEVERITY"] in ("CRITICAL", "HIGH", "MEDIUM")]

    if flagged:
        most_severe = min(flagged, key=lambda r: _SEVERITY_RANK.get(r["DRIFT_SEVERITY"], 9))
        severity_level = most_severe["DRIFT_SEVERITY"]
        detected = True
    else:
        most_severe = None
        severity_level = "NONE"
        detected = False

    confidence = _SEVERITY_CONFIDENCE.get(severity_level, 50.0)
    severity = _SEVERITY_TO_RESULT.get(severity_level, "low")

    n_analysed = len(all_results)
    reason = (
        f"{len(flagged)} of {n_analysed} analysed observation(s) flagged"
        if detected else
        f"No drift/anomalies detected in {n_analysed} analysed observation(s)"
    )
    logger.debug("Final result counts: %d flagged, %d normal/insufficient (of %d analysed)",
                 len(flagged), n_analysed - len(flagged), n_analysed)

    details = {
        "reason": reason,
        "results": all_results,
        "most_severe": most_severe,
        "validation_report": validation_report,
    }

    return LCSAnomalyResult(detected=detected, confidence=confidence, severity=severity,
                             details=details, visualizable=True)
