"""
matrix_spike_detector.py
========================
Production anomaly detection module for Matrix Spike (MS) QC samples.

Combines three analytical workstreams:
  - MS_Detection.ipynb        : feature engineering, rule-based flagging,
                                Isolation Forest, final risk scoring
  - MS_drift_detection1.ipynb : sustained directional drift detection across
                                historical Scheme-Analyte-Unit series
  - MS_Time_Series_Drift_Analysis.ipynb : rolling warning-scale behaviour

CLI usage
---------
    python matrix_spike_detector.py --input data/raw/QC_Anomaly_Training_Data_v2.xlsx

Callable from pipeline
----------------------
    from matrix_spike_detector import run_ms_detection, load_ms_config, MSConfig
    cfg = load_ms_config("config/matrix_spike_detector.yaml")
    results, drift_summary = run_ms_detection(df, cfg)
"""

from __future__ import annotations

import argparse
import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

try:
    from scipy.stats import theilslopes
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

MODEL_VERSION = "ms-detector-v1.0"
RULE_VERSION  = "ms-qc-rules-v1.0"

# ---------------------------------------------------------------------------
# Column sets
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "ANALYTICAL_TYPE", "QC_TYPE", "STD_LOT_CODE", "STD_CODE", "JOB_CODE",
    "NUMERIC_FINAL_VALUE", "ANALYSED_DATE", "SCHEME_CODE", "ANALYTE_CODE",
    "UNIT_CODE", "STANDARD_STATUS",
    "INTERNAL_MIN_VALUE", "INTERNAL_MAX_VALUE",
    "INTERNAL_MIN_WARNING_VALUE", "INTERNAL_MAX_WARNING_VALUE",
    "INTERNAL_TARGET_VALUE",
]

OUTPUT_COLUMNS = [
    "STD_LOT_CODE", "SCHEME_CODE", "ANALYTE_CODE", "JOB_CODE", "ANALYSED_DATE",
    "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE",
    "INTERNAL_MIN_VALUE", "INTERNAL_MAX_VALUE",
    "INTERNAL_MIN_WARNING_VALUE", "INTERNAL_MAX_WARNING_VALUE",
    "TARGET_DEVIATION", "TARGET_DEVIATION_PCT",
    "TRANSFORMED_VALUE",
    "NORM_DEV", "ABS_NORM_DEV",
    "POSITION_IN_RANGE",
    "DISTANCE_TO_UPPER_FAILURE", "DISTANCE_TO_LOWER_FAILURE",
    "ROLLING_MEAN_NORM_DEV", "ROLLING_STD_NORM_DEV", "ROLLING_BREACH_COUNT",
    "LAG_NORM_DEV", "ROBUST_Z_SCORE",
    "ROLLING_WARNING_POSITION",
    "RULE_FLAG", "RULE_FLAG_COLOUR", "RULE_FLAG_REASON",
    "DRIFT_FLAG", "DRIFT_DIRECTION", "DRIFT_ONSET",
    "IF_SCORE", "IF_ANOMALY", "IF_STATUS", "IF_MODEL",
    "FINAL_RISK", "DETECTION_METHOD", "REASON",
    "STANDARD_STATUS", "CCLAS_MISMATCH",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT        = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "matrix_spike_detector.yaml"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "ms_outputs"


@dataclass(frozen=True)
class MSConfig:
    # grouping
    group_cols: tuple = ("STD_LOT_CODE", "SCHEME_CODE", "ANALYTE_CODE")
    drift_group_cols: tuple = ("SCHEME_CODE", "ANALYTE_CODE", "UNIT_CODE")
    ignored_statuses: tuple = ("IgnoredUpperFailure", "IgnoredLowerFailure")

    # feature engineering
    rolling_window: int        = 10
    rolling_min_periods: int   = 3
    mad_scale: float           = 0.6745
    robust_z_threshold: float  = 3.0
    drift_threshold: float     = 0.5
    extreme_score_threshold: float = 10.0

    # drift detection (drift_detection1 approach)
    drift_rolling_window: int        = 10
    drift_consecutive_required: int  = 3
    drift_boundary_level: float      = 1.0
    drift_trend_window: int          = 10
    drift_slope_eps: float           = 1e-6

    # time-series (MS_Time_Series_Drift_Analysis approach)
    ts_rolling_window: int           = 10
    ts_min_periods: int              = 5
    ts_early_drift_fraction: float   = 0.75
    ts_persistence_windows: int      = 4
    ts_strong_drift_fraction: float  = 1.00
    ts_strong_persistence_windows: int = 3

    # isolation forest
    iforest_n_estimators: int    = 200
    iforest_contamination: float = 0.05
    iforest_random_state: int    = 42
    iforest_min_train: int       = 20

    # output
    output_dir: str = str(DEFAULT_OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _parse_config_scalar(value: str) -> object:
    raw = value.split("#", 1)[0].strip()
    if raw == "":
        return ""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw.strip("\"'")


def _parse_simple_yaml(path: Path) -> dict:
    root = {}
    stack = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, value = raw_line.strip().partition(":")
        if not separator:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_config_scalar(value)
    return root


def load_ms_config(path_str=None) -> MSConfig:
    """Load MSConfig from YAML, falling back to defaults if file absent."""
    path = Path(path_str or str(DEFAULT_CONFIG_PATH)).expanduser().resolve()
    if not path.exists():
        return MSConfig()
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        data = _parse_simple_yaml(path)
    if not isinstance(data, dict):
        return MSConfig()
    ms = data.get("matrix_spike", {})
    if not isinstance(ms, dict):
        return MSConfig()

    def _get(section, key, default):
        s = ms.get(section, {})
        return s.get(key, default) if isinstance(s, dict) else default

    return MSConfig(
        rolling_window=int(ms.get("rolling_window", MSConfig.rolling_window)),
        rolling_min_periods=int(ms.get("rolling_min_periods", MSConfig.rolling_min_periods)),
        mad_scale=float(ms.get("mad_scale", MSConfig.mad_scale)),
        robust_z_threshold=float(ms.get("robust_z_threshold", MSConfig.robust_z_threshold)),
        drift_threshold=float(ms.get("drift_threshold", MSConfig.drift_threshold)),
        extreme_score_threshold=float(ms.get("extreme_score_threshold", MSConfig.extreme_score_threshold)),
        drift_rolling_window=int(_get("drift", "rolling_window", MSConfig.drift_rolling_window)),
        drift_consecutive_required=int(_get("drift", "consecutive_required", MSConfig.drift_consecutive_required)),
        drift_boundary_level=float(_get("drift", "boundary_level", MSConfig.drift_boundary_level)),
        drift_trend_window=int(_get("drift", "trend_window", MSConfig.drift_trend_window)),
        ts_rolling_window=int(_get("time_series", "rolling_window", MSConfig.ts_rolling_window)),
        ts_min_periods=int(_get("time_series", "min_periods", MSConfig.ts_min_periods)),
        ts_early_drift_fraction=float(_get("time_series", "early_drift_fraction", MSConfig.ts_early_drift_fraction)),
        ts_persistence_windows=int(_get("time_series", "persistence_windows", MSConfig.ts_persistence_windows)),
        ts_strong_drift_fraction=float(_get("time_series", "strong_drift_fraction", MSConfig.ts_strong_drift_fraction)),
        ts_strong_persistence_windows=int(_get("time_series", "strong_persistence_windows", MSConfig.ts_strong_persistence_windows)),
        iforest_n_estimators=int(_get("isolation_forest", "n_estimators", MSConfig.iforest_n_estimators)),
        iforest_contamination=float(_get("isolation_forest", "contamination", MSConfig.iforest_contamination)),
        iforest_random_state=int(_get("isolation_forest", "random_state", MSConfig.iforest_random_state)),
        iforest_min_train=int(_get("isolation_forest", "min_train", MSConfig.iforest_min_train)),
        output_dir=str(ms.get("output_dir", MSConfig.output_dir)),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ms_data(path_str: str, sheet_name: str = "SPK(MS) Assessment") -> pd.DataFrame:
    """Load and column-normalise an MS data file (CSV or Excel)."""
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, low_memory=False, keep_default_na=False)
    else:
        raw = pd.read_excel(path, sheet_name=sheet_name, keep_default_na=False, engine="openpyxl")
    raw = raw.dropna(axis=1, how="all")
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    unnamed = [c for c in raw.columns if c.strip() == "" or c.startswith("UNNAMED")]
    if unnamed:
        raw = raw.drop(columns=unnamed)
    return raw


def filter_ms_records(raw: pd.DataFrame) -> pd.DataFrame:
    """Filter to Spike/MS records; exclude ANALYTE_CODE='NA' (non-analyte artefact)."""
    for col in ["ANALYTICAL_TYPE", "QC_TYPE", "ANALYTE_CODE"]:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip()
    mask = (
        raw["ANALYTICAL_TYPE"].str.casefold().eq("spike")
        & raw["QC_TYPE"].str.casefold().eq("ms")
        & raw["ANALYTE_CODE"].ne("NA")
    )
    return raw.loc[mask].copy()


def validate_ms_frame(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"MS detector: missing required columns: {missing}")


def prepare_ms_records(df: pd.DataFrame, cfg: MSConfig) -> pd.DataFrame:
    """Coerce types, mark ignored records, sort chronologically."""
    numeric_cols = [
        "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE",
        "INTERNAL_MIN_VALUE", "INTERNAL_MAX_VALUE",
        "INTERNAL_MIN_WARNING_VALUE", "INTERNAL_MAX_WARNING_VALUE",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ANALYSED_DATE"] = pd.to_datetime(df["ANALYSED_DATE"], errors="coerce")
    for col in ["STD_LOT_CODE", "STD_CODE", "JOB_CODE",
                "SCHEME_CODE", "ANALYTE_CODE", "UNIT_CODE", "STANDARD_STATUS"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["IS_IGNORED"] = df["STANDARD_STATUS"].isin(cfg.ignored_statuses)
    df = df.sort_values(list(cfg.group_cols) + ["ANALYSED_DATE"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, cfg: MSConfig) -> pd.DataFrame:
    gc   = list(cfg.group_cols)
    rw   = cfg.rolling_window
    rmin = cfg.rolling_min_periods

    # 1-2. Deviation from target
    df["TARGET_DEVIATION"] = df["NUMERIC_FINAL_VALUE"] - df["INTERNAL_TARGET_VALUE"]
    df["TARGET_DEVIATION_PCT"] = np.where(
        df["INTERNAL_TARGET_VALUE"].abs() > 1e-9,
        (df["TARGET_DEVIATION"] / df["INTERNAL_TARGET_VALUE"]) * 100,
        np.nan,
    )

    # 3-4. Normalised deviation (0=target, +-1=warning boundary)
    upper_span = df["INTERNAL_MAX_WARNING_VALUE"] - df["INTERNAL_TARGET_VALUE"]
    lower_span = df["INTERNAL_TARGET_VALUE"] - df["INTERNAL_MIN_WARNING_VALUE"]
    df["NORM_DEV"] = np.where(
        df["NUMERIC_FINAL_VALUE"] >= df["INTERNAL_TARGET_VALUE"],
        np.where(upper_span.abs() > 1e-9, df["TARGET_DEVIATION"] / upper_span, np.nan),
        np.where(lower_span.abs() > 1e-9, df["TARGET_DEVIATION"] / lower_span, np.nan),
    )
    df["ABS_NORM_DEV"] = df["NORM_DEV"].abs()

    # 5-8. Limit position
    span = df["INTERNAL_MAX_VALUE"] - df["INTERNAL_MIN_VALUE"]
    df["DISTANCE_TO_UPPER_FAILURE"] = df["INTERNAL_MAX_VALUE"] - df["NUMERIC_FINAL_VALUE"]
    df["DISTANCE_TO_LOWER_FAILURE"] = df["NUMERIC_FINAL_VALUE"] - df["INTERNAL_MIN_VALUE"]
    df["LIMIT_SPAN"]        = span
    df["POSITION_IN_RANGE"] = np.where(
        span.abs() > 1e-9,
        (df["NUMERIC_FINAL_VALUE"] - df["INTERNAL_MIN_VALUE"]) / span,
        np.nan,
    )

    # 9-11. Rolling temporal features
    df["ROLLING_MEAN_NORM_DEV"] = df.groupby(gc)["NORM_DEV"].transform(
        lambda x: x.rolling(rw, rmin).mean()
    )
    df["ROLLING_STD_NORM_DEV"] = df.groupby(gc)["NORM_DEV"].transform(
        lambda x: x.rolling(rw, rmin).std()
    )
    df["ROLLING_BREACH_COUNT"] = df.groupby(gc)["NORM_DEV"].transform(
        lambda x: (x.abs() > 1).astype(float).rolling(rw, rmin).sum()
    )

    # 12-13. Group baseline (non-ignored only)
    baseline = (
        df[~df["IS_IGNORED"]].groupby(gc)["NORM_DEV"]
        .median().rename("GROUP_MEDIAN")
    )
    df = df.join(baseline, on=gc)
    df["ABS_DEV_FROM_MEDIAN"] = (df["NORM_DEV"] - df["GROUP_MEDIAN"]).abs()
    group_mad = (
        df[~df["IS_IGNORED"]].groupby(gc)["ABS_DEV_FROM_MEDIAN"]
        .median().rename("GROUP_MAD")
    )
    df = df.join(group_mad, on=gc)

    # 14. Robust Z-score
    df["ROBUST_Z_SCORE"] = np.where(
        df["GROUP_MAD"] > 1e-9,
        cfg.mad_scale * (df["NORM_DEV"] - df["GROUP_MEDIAN"]) / df["GROUP_MAD"],
        np.nan,
    )

    # 15. Lag feature
    df["LAG_NORM_DEV"] = df.groupby(gc)["NORM_DEV"].transform(lambda x: x.shift(1))

    return df


# ---------------------------------------------------------------------------
# Transformed value (rule-based layer uses relative scale)
# ---------------------------------------------------------------------------

def transform_ms_values(df: pd.DataFrame) -> pd.DataFrame:
    """Piecewise transform: TARGET=0, FAILURE limits=+-1, warnings proportional."""
    target     = df["INTERNAL_TARGET_VALUE"]
    upper_span = df["INTERNAL_MAX_VALUE"] - target
    lower_span = target - df["INTERNAL_MIN_VALUE"]
    value      = df["NUMERIC_FINAL_VALUE"]
    span_for_value = pd.Series(
        np.where(value >= target, upper_span, lower_span), index=df.index
    )
    df["TRANSFORMED_VALUE"] = np.where(
        span_for_value.abs() > 1e-9, (value - target) / span_for_value, np.nan
    )
    df["TRANSFORMED_TARGET"]      = 0.0
    df["TRANSFORMED_MAX"]         = 1.0
    df["TRANSFORMED_MIN"]         = -1.0
    df["TRANSFORMED_MAX_WARNING"] = np.where(
        upper_span.abs() > 1e-9,
        (df["INTERNAL_MAX_WARNING_VALUE"] - target) / upper_span, np.nan
    )
    df["TRANSFORMED_MIN_WARNING"] = np.where(
        lower_span.abs() > 1e-9,
        (df["INTERNAL_MIN_WARNING_VALUE"] - target) / lower_span, np.nan
    )
    return df


# ---------------------------------------------------------------------------
# Rule-based flag layer
# ---------------------------------------------------------------------------

def _inclusive(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[col].fillna("Y").astype(str).str.strip().str.upper().eq("Y")


def apply_rule_based_flags(df: pd.DataFrame) -> pd.DataFrame:
    val      = df["TRANSFORMED_VALUE"]
    max_warn = df["TRANSFORMED_MAX_WARNING"]
    min_warn = df["TRANSFORMED_MIN_WARNING"]
    is_ign   = df["IS_IGNORED"]

    above_max_fail = np.where(_inclusive(df, "INTERNAL_MAX_INCLUSIVE"), val >= 1.0, val > 1.0)
    below_min_fail = np.where(_inclusive(df, "INTERNAL_MIN_INCLUSIVE"), val <= -1.0, val < -1.0)
    is_failure     = above_max_fail | below_min_fail

    above_max_warn = np.where(_inclusive(df, "INTERNAL_MAX_WARNING_INCLUSIVE"), val >= max_warn, val > max_warn)
    below_min_warn = np.where(_inclusive(df, "INTERNAL_MIN_WARNING_INCLUSIVE"), val <= min_warn, val < min_warn)
    is_warning     = (~is_failure) & (above_max_warn | below_min_warn)

    df["RULE_FLAG"] = np.select(
        [is_ign, is_failure, is_warning], ["Ignored", "Failure", "Warning"], default="Pass"
    )
    df["RULE_FLAG_COLOUR"] = np.select(
        [is_ign, is_failure, is_warning], ["grey", "red", "yellow"], default="green"
    )
    df["RULE_FLAG_REASON"] = np.select(
        [
            is_ign,
            below_min_fail & ~is_ign,
            above_max_fail & ~is_ign,
            below_min_warn & ~is_failure & ~is_ign,
            above_max_warn & ~is_failure & ~is_ign,
        ],
        [
            "Result manually ignored in CCLAS",
            "Transformed value (" + val.round(4).astype(str) + ") below lower failure boundary (-1.0) — LowerFailure",
            "Transformed value (" + val.round(4).astype(str) + ") above upper failure boundary (1.0) — UpperFailure",
            "Transformed value (" + val.round(4).astype(str) + ") below lower warning boundary (" + min_warn.round(4).astype(str) + ") — LowerWarning",
            "Transformed value (" + val.round(4).astype(str) + ") above upper warning boundary (" + max_warn.round(4).astype(str) + ") — UpperWarning",
        ],
        default="Transformed value (" + val.round(4).astype(str) + ") within warning limits — Pass",
    )
    return df


# ---------------------------------------------------------------------------
# Time-series rolling warning position (MS_Time_Series_Drift_Analysis)
# ---------------------------------------------------------------------------

def _directional_normalise(deviation, target, lower_col, upper_col):
    upper_dist = upper_col - target
    lower_dist = target - lower_col
    result     = pd.Series(np.nan, index=deviation.index, dtype=float)
    upper_mask = (deviation >= 0) & (upper_dist > 0)
    lower_mask = (deviation < 0) & (lower_dist > 0)
    result.loc[upper_mask] = deviation.loc[upper_mask] / upper_dist.loc[upper_mask]
    result.loc[lower_mask] = deviation.loc[lower_mask] / lower_dist.loc[lower_mask]
    return result.replace([np.inf, -np.inf], np.nan)


def compute_rolling_warning_position(df: pd.DataFrame, cfg: MSConfig) -> pd.DataFrame:
    """Rolling mean of warning-scale position within each Scheme-Analyte-Unit group."""
    dg = list(cfg.drift_group_cols)
    df["DEVIATION"] = df["NUMERIC_FINAL_VALUE"] - df["INTERNAL_TARGET_VALUE"]
    df["WARNING_SCALE_POSITION"] = _directional_normalise(
        df["DEVIATION"], df["INTERNAL_TARGET_VALUE"],
        df["INTERNAL_MIN_WARNING_VALUE"], df["INTERNAL_MAX_WARNING_VALUE"],
    )
    df["ROLLING_WARNING_POSITION"] = df.groupby(dg)["WARNING_SCALE_POSITION"].transform(
        lambda x: x.rolling(cfg.ts_rolling_window, cfg.ts_min_periods).mean()
    )
    return df


# ---------------------------------------------------------------------------
# Drift detection (MS_drift_detection1)
# ---------------------------------------------------------------------------

def _first_consecutive_directional_breach(values, dates, threshold, consecutive_required):
    run_direction = None
    run_start     = pd.NaT
    run_len       = 0
    for value, date in zip(values, dates):
        if pd.isna(value) or abs(value) < threshold:
            run_direction = None
            run_start     = pd.NaT
            run_len       = 0
            continue
        direction = "Upper" if value > 0 else "Lower"
        if direction == run_direction:
            run_len += 1
        else:
            run_direction = direction
            run_start     = date
            run_len       = 1
        if run_len >= consecutive_required:
            return True, run_direction, run_start
    return False, None, pd.NaT


def _theil_sen_slope(values, window, slope_eps):
    if not _SCIPY_AVAILABLE:
        return np.nan, "FLAT", np.nan
    recent = [v for v in values[-window:] if not np.isnan(v)]
    if len(recent) < 2:
        return np.nan, "FLAT", np.nan
    x = np.arange(len(recent), dtype=float)
    y = np.array(recent)
    slope, *_ = theilslopes(y, x)
    direction  = "UP" if slope > slope_eps else ("DOWN" if slope < -slope_eps else "FLAT")
    diffs      = np.diff(y)
    nonzero    = diffs[diffs != 0]
    strength   = float((np.sign(nonzero) == np.sign(slope)).mean()) if len(nonzero) > 0 and slope != 0 else 0.0
    return float(slope), direction, strength


def _evaluate_drift_series(group: pd.DataFrame, cfg: MSConfig) -> dict:
    g   = group.sort_values("ANALYSED_DATE").reset_index(drop=True).copy()
    win = min(cfg.drift_rolling_window, len(g))
    mp  = max(1, win // 2)

    g["ROLLING_WARN"] = g["WARNING_SCALE_POSITION"].rolling(window=win, min_periods=mp).mean()
    warn_detected, warn_dir, warn_start = _first_consecutive_directional_breach(
        g["ROLLING_WARN"], g["ANALYSED_DATE"],
        cfg.drift_boundary_level, cfg.drift_consecutive_required,
    )

    failure_scale = _directional_normalise(
        g["DEVIATION"], g["INTERNAL_TARGET_VALUE"],
        g["INTERNAL_MIN_VALUE"], g["INTERNAL_MAX_VALUE"],
    )
    g["ROLLING_FAIL"] = failure_scale.rolling(window=win, min_periods=mp).mean()
    if warn_detected:
        fail_detected, fail_dir, fail_start = _first_consecutive_directional_breach(
            g["ROLLING_FAIL"], g["ANALYSED_DATE"],
            cfg.drift_boundary_level, cfg.drift_consecutive_required,
        )
    else:
        fail_detected, fail_dir, fail_start = False, None, pd.NaT

    ts_slope, ts_dir, ts_strength = _theil_sen_slope(
        g["WARNING_SCALE_POSITION"].tolist(), cfg.drift_trend_window, cfg.drift_slope_eps
    )

    status     = g["STANDARD_STATUS"].astype(str)
    warn_mask  = status.str.contains("Warning", case=False) & ~status.str.contains("Ignored", case=False)
    fail_mask  = status.str.contains("Failure", case=False) & ~status.str.contains("Ignored", case=False)
    cclas_warn = g.loc[warn_mask, "ANALYSED_DATE"].min() if warn_mask.any() else pd.NaT
    cclas_fail = g.loc[fail_mask, "ANALYSED_DATE"].min() if fail_mask.any() else pd.NaT

    valid_warn_roll = g["ROLLING_WARN"].dropna()
    valid_fail_roll = g["ROLLING_FAIL"].dropna()

    return {
        "warning_drift_detected":    bool(warn_detected),
        "warning_direction":         warn_dir,
        "warning_drift_start":       warn_start,
        "failure_drift_detected":    bool(fail_detected),
        "failure_direction":         fail_dir,
        "failure_drift_start":       fail_start,
        "current_warning_score":     float(valid_warn_roll.iloc[-1]) if not valid_warn_roll.empty else np.nan,
        "peak_abs_warning_score":    float(valid_warn_roll.abs().max()) if not valid_warn_roll.empty else np.nan,
        "peak_abs_failure_score":    float(valid_fail_roll.abs().max()) if not valid_fail_roll.empty else np.nan,
        "theil_sen_slope":           ts_slope,
        "theil_sen_direction":       ts_dir,
        "theil_sen_strength":        ts_strength,
        "first_status_warning_date": cclas_warn,
        "first_status_failure_date": cclas_fail,
        "n_observations":            len(g),
        "n_jobs":                    g["JOB_CODE"].nunique(),
        "n_extreme_outliers":        int((g["WARNING_SCALE_POSITION"].abs() > cfg.extreme_score_threshold).sum()),
    }


def run_drift_detection(df: pd.DataFrame, cfg: MSConfig) -> pd.DataFrame:
    """Run drift detection across all Scheme-Analyte-Unit groups."""
    dg   = list(cfg.drift_group_cols)
    rows = []
    for keys, group in df.groupby(dg, sort=True):
        result = _evaluate_drift_series(group, cfg)
        row    = dict(zip(dg, keys))
        row.update(result)
        rows.append(row)
    drift = pd.DataFrame(rows)
    if drift.empty:
        return drift
    drift["drift_level"] = np.select(
        [drift["failure_drift_detected"], drift["warning_drift_detected"]],
        ["Failure", "Warning"],
        default="None",
    )
    return (
        drift.sort_values(
            ["failure_drift_detected", "warning_drift_detected", "peak_abs_warning_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    )


def _merge_drift_flags(df: pd.DataFrame, drift_summary: pd.DataFrame, cfg: MSConfig) -> pd.DataFrame:
    dg = list(cfg.drift_group_cols)
    if drift_summary.empty:
        df["DRIFT_FLAG"]      = "None"
        df["DRIFT_DIRECTION"] = ""
        df["DRIFT_ONSET"]     = pd.NaT
        return df
    slim = drift_summary[dg + ["drift_level", "warning_direction", "warning_drift_start"]].rename(columns={
        "drift_level":         "DRIFT_FLAG",
        "warning_direction":   "DRIFT_DIRECTION",
        "warning_drift_start": "DRIFT_ONSET",
    })
    df = df.merge(slim, on=dg, how="left")
    df["DRIFT_FLAG"]      = df["DRIFT_FLAG"].fillna("None")
    df["DRIFT_DIRECTION"] = df["DRIFT_DIRECTION"].fillna("")
    return df


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

IF_FEATURES = [
    "NORM_DEV", "ABS_NORM_DEV", "TRANSFORMED_VALUE", "POSITION_IN_RANGE",
    "ROLLING_MEAN_NORM_DEV", "ROLLING_STD_NORM_DEV", "ROLLING_BREACH_COUNT",
    "ROBUST_Z_SCORE", "DISTANCE_TO_UPPER_FAILURE", "DISTANCE_TO_LOWER_FAILURE",
    "LAG_NORM_DEV",
]


def run_isolation_forest(df: pd.DataFrame, cfg: MSConfig) -> pd.DataFrame:
    """Per-analyte Isolation Forest with global fallback."""
    train_mask = ~df["IS_IGNORED"]
    X_all      = df[IF_FEATURES].copy()
    imputer    = SimpleImputer(strategy="median")
    imputer.fit(X_all[train_mask])
    X_imp      = pd.DataFrame(imputer.transform(X_all), columns=IF_FEATURES, index=df.index)

    global_if = IsolationForest(
        n_estimators=cfg.iforest_n_estimators,
        contamination=cfg.iforest_contamination,
        random_state=cfg.iforest_random_state,
        n_jobs=-1,
    )
    global_if.fit(X_imp[train_mask])

    if_scores  = pd.Series(np.nan,   index=df.index)
    if_anomaly = pd.Series(False,    index=df.index)
    if_model   = pd.Series("global", index=df.index)

    for (analyte, scheme), group_idx in df.groupby(["ANALYTE_CODE", "SCHEME_CODE"]).groups.items():
        train_idx = group_idx[train_mask[group_idx]]
        n_train   = len(train_idx)
        X_group   = X_imp.loc[group_idx]

        if n_train >= cfg.iforest_min_train:
            g_if = IsolationForest(
                n_estimators=cfg.iforest_n_estimators,
                contamination=cfg.iforest_contamination,
                random_state=cfg.iforest_random_state,
                n_jobs=-1,
            )
            g_if.fit(X_imp.loc[train_idx])
            if_scores[group_idx]  = -g_if.score_samples(X_group)
            if_anomaly[group_idx] = g_if.predict(X_group) == -1
            if_model[group_idx]   = "per-analyte"
        else:
            if_scores[group_idx]  = -global_if.score_samples(X_group)
            if_anomaly[group_idx] = global_if.predict(X_group) == -1
            if_model[group_idx]   = f"global (n={n_train}<{cfg.iforest_min_train})"

    df["IF_SCORE"]   = if_scores
    df["IF_ANOMALY"] = if_anomaly
    df["IF_STATUS"]  = np.where(df["IF_ANOMALY"], "WARNING", "PASS")
    df["IF_MODEL"]   = if_model
    return df


# ---------------------------------------------------------------------------
# Final risk scoring
# ---------------------------------------------------------------------------

def _assign_final_risk(rule_flag: str, if_anomaly: bool, drift_flag: str, is_ignored: bool) -> str:
    if is_ignored:
        return "Ignored"
    if rule_flag == "Failure":
        return "Critical"
    if rule_flag == "Warning" and if_anomaly:
        return "High"
    if rule_flag == "Warning":
        return "Medium"
    if rule_flag == "Pass" and if_anomaly:
        return "Medium"
    return "Low"


def _build_reason(
    rule_reason, rule_flag, if_anomaly, drift_flag,
    robust_z, rolling_mean, rolling_breach, is_ignored, cfg: MSConfig,
) -> str:
    if is_ignored:
        return rule_reason
    parts = [rule_reason]
    try:
        if not np.isnan(float(robust_z or "nan")) and abs(float(robust_z)) > cfg.robust_z_threshold:
            parts.append(f"Robust Z-score {float(robust_z):.2f} exceeds threshold (|Z|>{cfg.robust_z_threshold})")
    except (TypeError, ValueError):
        pass
    try:
        if not np.isnan(float(rolling_mean or "nan")) and abs(float(rolling_mean)) > cfg.drift_threshold:
            direction = "above" if float(rolling_mean) > 0 else "below"
            parts.append(f"Rolling mean {float(rolling_mean):.2f} sustained {direction} drift threshold (+-{cfg.drift_threshold})")
    except (TypeError, ValueError):
        pass
    try:
        if not np.isnan(float(rolling_breach or "nan")) and float(rolling_breach) >= 5:
            parts.append(f"{int(rolling_breach)} of last {cfg.rolling_window} observations outside warning boundary")
    except (TypeError, ValueError):
        pass
    if drift_flag in ("Warning", "Failure"):
        parts.append(f"Historical series drift detected — drift level: {drift_flag}")
    if if_anomaly and rule_flag == "Pass":
        parts.append("Isolation Forest flagged as multivariate anomaly despite passing rule-based checks")
    return " | ".join(parts)


def apply_risk_scoring(df: pd.DataFrame, cfg: MSConfig) -> pd.DataFrame:
    df["FINAL_RISK"] = [
        _assign_final_risk(row["RULE_FLAG"], row["IF_ANOMALY"], row.get("DRIFT_FLAG", "None"), row["IS_IGNORED"])
        for _, row in df.iterrows()
    ]
    df["DETECTION_METHOD"] = np.select(
        [
            df["IS_IGNORED"],
            df["RULE_FLAG"] == "Failure",
            (df["RULE_FLAG"] == "Warning") & df["IF_ANOMALY"],
            df["RULE_FLAG"] == "Warning",
            (df["RULE_FLAG"] == "Pass") & df["IF_ANOMALY"],
        ],
        ["Ignored", "Rule-based", "Rule-based + IsolationForest", "Rule-based", "IsolationForest"],
        default="None",
    )
    df["REASON"] = [
        _build_reason(
            row["RULE_FLAG_REASON"], row["RULE_FLAG"], row["IF_ANOMALY"],
            row.get("DRIFT_FLAG", "None"), row.get("ROBUST_Z_SCORE"),
            row.get("ROLLING_MEAN_NORM_DEV"), row.get("ROLLING_BREACH_COUNT"),
            row["IS_IGNORED"], cfg,
        )
        for _, row in df.iterrows()
    ]
    return df


# ---------------------------------------------------------------------------
# CCLAS validation
# ---------------------------------------------------------------------------

def validate_against_cclas(df: pd.DataFrame) -> pd.DataFrame:
    df["CCLAS_STATUS_NORMALISED"] = np.select(
        [
            df["STANDARD_STATUS"].isin(["UpperFailure", "LowerFailure"]),
            df["STANDARD_STATUS"].isin(["UpperWarning", "LowerWarning"]),
            df["STANDARD_STATUS"].isin(["IgnoredUpperFailure", "IgnoredLowerFailure"]),
        ],
        ["FAIL", "WARNING", "IGNORED"],
        default="PASS",
    )
    df["RULE_FLAG_NORMALISED"] = df["RULE_FLAG"].str.upper().replace({"FAILURE": "FAIL"})
    df["CCLAS_MISMATCH"] = (
        ~df["IS_IGNORED"] & (df["RULE_FLAG_NORMALISED"] != df["CCLAS_STATUS_NORMALISED"])
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ms_detection(
    df: pd.DataFrame,
    cfg: MSConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full MS anomaly detection pipeline.

    Parameters
    ----------
    df  : Filtered MS DataFrame (ANALYTICAL_TYPE=Spike, QC_TYPE=MS).
    cfg : MSConfig. Defaults to MSConfig() if None.

    Returns
    -------
    results       : per-record anomaly output DataFrame
    drift_summary : per-Scheme-Analyte-Unit drift summary DataFrame
    """
    if cfg is None:
        cfg = MSConfig()

    validate_ms_frame(df)
    df = prepare_ms_records(df.copy(), cfg)
    df = engineer_features(df, cfg)
    df = transform_ms_values(df)
    df = compute_rolling_warning_position(df, cfg)
    df = apply_rule_based_flags(df)

    drift_summary = run_drift_detection(df, cfg)
    df = _merge_drift_flags(df, drift_summary, cfg)

    df = run_isolation_forest(df, cfg)
    df = apply_risk_scoring(df, cfg)
    df = validate_against_cclas(df)

    out_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    results  = df[out_cols].copy()

    return results, drift_summary


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_results(
    results: pd.DataFrame,
    drift_summary: pd.DataFrame,
    output_dir,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    full_path = output_dir / "ms_detection_results.csv"
    results.sort_values("IF_SCORE", ascending=False).to_csv(full_path, index=False)
    paths["full"] = full_path

    flagged = results[results["FINAL_RISK"].isin(["Critical", "High", "Medium"])].copy()
    flagged_path = output_dir / "ms_detection_flagged.csv"
    flagged.sort_values("IF_SCORE", ascending=False).to_csv(flagged_path, index=False)
    paths["flagged"] = flagged_path

    if not drift_summary.empty:
        drift_path = output_dir / "ms_drift_results.csv"
        drift_summary.to_csv(drift_path, index=False)
        paths["drift"] = drift_path

        drift_flagged = drift_summary[drift_summary["drift_level"] != "None"].copy()
        drift_flagged.to_csv(output_dir / "ms_drift_flagged.csv", index=False)
        paths["drift_flagged"] = output_dir / "ms_drift_flagged.csv"

    return paths


def print_summary(results: pd.DataFrame, drift_summary: pd.DataFrame) -> None:
    risk_counts = results["FINAL_RISK"].value_counts()
    print("\nMS Detection Summary")
    print("=" * 40)
    print(f"Total records:     {len(results):,}")
    for level in ["Critical", "High", "Medium", "Low", "Ignored"]:
        print(f"  {level:<12}  {risk_counts.get(level, 0):>5}")
    print(f"\nIF anomalies:      {results['IF_ANOMALY'].sum():,}")
    print(f"CCLAS mismatches:  {results['CCLAS_MISMATCH'].sum():,}")
    if not drift_summary.empty:
        dc = drift_summary["drift_level"].value_counts()
        print(f"\nDrift series:      {len(drift_summary):,}")
        for level in ["Failure", "Warning", "None"]:
            print(f"  {level:<10}  {dc.get(level, 0):>4}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Matrix Spike anomaly detection pipeline.")
    parser.add_argument("--input", required=True, help="Path to CSV or Excel workbook.")
    parser.add_argument("--sheet", default="SPK(MS) Assessment", help="Sheet name for Excel inputs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--contamination", type=float, default=None)
    parser.add_argument("--rolling-window", type=int, default=None)
    parser.add_argument("--drift-threshold", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_ms_config(args.config)

    overrides = {}
    if args.contamination is not None:
        overrides["iforest_contamination"] = min(max(args.contamination, 0.001), 0.2)
    if args.rolling_window is not None:
        overrides["rolling_window"] = args.rolling_window
    if args.drift_threshold is not None:
        overrides["drift_threshold"] = args.drift_threshold
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    cfg = dataclasses.replace(cfg, output_dir=args.output_dir)

    print(f"Loading MS data from: {args.input}")
    raw = load_ms_data(args.input, sheet_name=args.sheet)
    ms  = filter_ms_records(raw)
    print(f"MS records after filter: {len(ms):,}")
    print(f"Unique analytes (NA excluded): {ms['ANALYTE_CODE'].nunique():,}")

    results, drift_summary = run_ms_detection(ms, cfg)
    print_summary(results, drift_summary)

    paths = export_results(results, drift_summary, cfg.output_dir)
    print("\nOutputs written:")
    for name, path in paths.items():
        print(f"  {name:<15} {path}")


if __name__ == "__main__":
    main()
