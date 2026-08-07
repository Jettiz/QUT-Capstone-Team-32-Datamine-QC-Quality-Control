"""
Control (Laboratory Control Sample) Anomaly Detector

Detects deviations in control samples that verify method accuracy and stability
using a clean, well-defined matrix with known analyte concentrations.

Key Characteristics:
- Known target value (mean)
- Acceptable tolerance window
- Long time-series per analyte and method

Anomaly Patterns to Detect:
1. Drift: systematic change in normalized deviation over time
   - Detected via rolling mean of normalized deviations
   - Flagged when exceeding threshold for consecutive observations

Method (from existing analysis):
- Normalize each value against warning bounds (range 0 to ±1)
- Calculate rolling mean of normalized deviations
- Flag drift when rolling mean exceeds DRIFT_THRESHOLD
  for at least CONSECUTIVE_REQUIRED consecutive observations

Configuration: config/lcs_config.yaml
- rolling_window: observations for rolling mean (default 10)
- drift_threshold: fraction of warning bound span (default 0.5)
- consecutive_required: breaches needed to flag drift (default 3)
- min_samples: minimum observations needed (default 10)
- sensitivity: high/medium/low
"""

import pandas as pd
import numpy as np
from .base_detector import LCSAnomalyResult
from typing import Dict, Any, Tuple, Optional
import yaml


class ControlDetector:
    """
    Detect anomalies in Control (Laboratory Control Sample) data.

    Method: Normalized Deviation with Rolling Mean
    - Normalizes measurements against warning bounds
    - Detects drift via rolling mean and threshold breaches
    - Identifies drift direction (Upper/Lower) and timing

    Methods:
    - detect(df, std_lot_code, job_code) -> LCSAnomalyResult
    - detect_drift(df) -> Tuple[pd.DataFrame, Dict]
    """

    def __init__(self, config_path: str):
        """
        Initialize detector with configuration.

        Args:
            config_path: Path to lcs_config.yaml
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def detect(self, df: pd.DataFrame, std_lot_code: str, job_code: str) -> LCSAnomalyResult:
        """
        Run complete LCS drift detection for a specific standard lot and job.

        Args:
            df: DataFrame with Standard records
            std_lot_code: Standard lot code to analyze
            job_code: Job code to analyze

        Returns:
            LCSAnomalyResult with drift status, severity, and details
        """
        rolling_window = self.config.get('rolling_window', 10)
        drift_threshold = self.config.get('drift_threshold', 0.5)
        consecutive_required = self.config.get('consecutive_required', 3)
        min_samples = self.config.get('min_samples', 10)

        summary, analyte_data = self.detect_drift(
            df,
            std_lot_code=std_lot_code,
            job_code=job_code,
            rolling_window=rolling_window,
            drift_threshold=drift_threshold,
            consecutive_required=consecutive_required,
        )

        if summary.empty:
            return LCSAnomalyResult(
                detected=False,
                confidence=0,
                severity="low",
                details={"reason": "No LCS data found for this lot/job"},
                visualizable=False,
            )

        # Summarize across all analytes
        n_drifting = summary["drift_detected"].sum()
        detected = n_drifting > 0

        if detected:
            # Get most severe drift
            drifting = summary[summary["drift_detected"]].copy()
            most_severe = drifting.loc[drifting["severity_score"].idxmax()]
            confidence = min(100, most_severe["severity_score"] * 100)
            severity = "high" if confidence >= 75 else "medium"

            details = {
                "reason": f"{n_drifting} analytes show drift in {std_lot_code}/{job_code}",
                "analytes_drifting": drifting["ANALYTE_CODE"].tolist(),
                "summary_table": summary.to_dict(orient='records'),
                "most_severe_analyte": most_severe["ANALYTE_CODE"],
                "most_severe_direction": most_severe["drift_direction"],
                "most_severe_start": most_severe["drift_start"].isoformat() if pd.notna(most_severe["drift_start"]) else None,
                "most_severe_score": round(most_severe["severity_score"], 4),
            }
        else:
            confidence = 95
            severity = "low"
            details = {
                "reason": f"No drift detected in {len(summary)} analytes",
                "summary_table": summary.to_dict(orient='records'),
            }

        return LCSAnomalyResult(
            detected=detected,
            confidence=confidence,
            severity=severity,
            details=details,
            visualizable=True,
        )

    def detect_drift(self, df: pd.DataFrame,
                     std_lot_code: str,
                     job_code: str,
                     rolling_window: int = 10,
                     drift_threshold: float = 0.5,
                     consecutive_required: int = 3) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect drift for all analytes within a given STD_LOT_CODE and JOB_CODE.

        Algorithm:
        1. Filter to Standard records for the lot/job
        2. For each analyte, sort by date
        3. Normalize each value against warning bounds:
           - If value >= target: norm_dev = (value - target) / (upper_warn - target)
           - If value < target: norm_dev = (value - target) / (target - lower_warn)
        4. Calculate rolling mean of norm_dev
        5. Flag drift when |rolling_mean| > drift_threshold for consecutive_required observations

        Returns:
            summary: DataFrame with one row per analyte showing drift results
            analyte_data: Dict mapping analyte to its detailed per-observation data
        """
        required_cols = [
            "ANALYTICAL_TYPE", "STD_LOT_CODE", "ANALYTE_CODE", "ANALYSED_DATE",
            "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE",
            "INTERNAL_MAX_WARNING_VALUE", "INTERNAL_MIN_WARNING_VALUE",
        ]

        lot_df = (
            df[
                (df["ANALYTICAL_TYPE"] == "Standard") &
                (df["STD_LOT_CODE"] == std_lot_code) &
                (df["JOB_CODE"] == job_code)
            ]
            .dropna(subset=required_cols)
            .copy()
        )

        if lot_df.empty:
            return pd.DataFrame(), {}

        summary_rows = []
        analyte_data = {}

        for analyte, adf in lot_df.groupby("ANALYTE_CODE"):
            adf = adf.sort_values("ANALYSED_DATE").reset_index(drop=True)

            target = adf["INTERNAL_TARGET_VALUE"].iloc[0]
            max_warn = adf["INTERNAL_MAX_WARNING_VALUE"].iloc[0]
            min_warn = adf["INTERNAL_MIN_WARNING_VALUE"].iloc[0]
            value = adf["NUMERIC_FINAL_VALUE"]

            upper_range = max_warn - target
            lower_range = target - min_warn

            # Normalize deviation: 0 = target, ±1 = warning bound
            adf["norm_dev"] = np.where(
                value >= target,
                (value - target) / upper_range if upper_range != 0 else 0,
                (value - target) / lower_range if lower_range != 0 else 0,
            )

            # Rolling mean
            win = min(rolling_window, len(adf))
            adf["rolling_norm_dev"] = (
                adf["norm_dev"]
                .rolling(window=win, min_periods=max(1, win // 2))
                .mean()
            )

            # Detect drift: consecutive breaches of threshold
            drift_detected = False
            drift_direction = None
            drift_start = pd.NaT
            run_len = 0
            run_start_idx = None

            for i, rnd in enumerate(adf["rolling_norm_dev"]):
                if pd.isna(rnd):
                    run_len = 0
                    run_start_idx = None
                    continue

                if abs(rnd) >= drift_threshold:
                    if run_len == 0:
                        run_start_idx = i
                    run_len += 1
                    if run_len >= consecutive_required and not drift_detected:
                        drift_detected = True
                        drift_direction = "Upper" if rnd > 0 else "Lower"
                        drift_start = adf.loc[run_start_idx, "ANALYSED_DATE"]
                else:
                    run_len = 0
                    run_start_idx = None

            last_rnd = adf["rolling_norm_dev"].dropna()
            severity_score = float(abs(last_rnd.iloc[-1])) if not last_rnd.empty else np.nan

            # First warning/failure dates
            first_warning_date = pd.NaT
            first_failure_date = pd.NaT
            if "STANDARD_STATUS" in adf.columns:
                warn_mask = (
                    adf["STANDARD_STATUS"].str.contains("Warning", na=False) &
                    ~adf["STANDARD_STATUS"].str.contains("Ignored", na=False)
                )
                fail_mask = (
                    adf["STANDARD_STATUS"].str.contains("Failure", na=False) &
                    ~adf["STANDARD_STATUS"].str.contains("Ignored", na=False)
                )
                if warn_mask.any():
                    first_warning_date = adf.loc[warn_mask, "ANALYSED_DATE"].min()
                if fail_mask.any():
                    first_failure_date = adf.loc[fail_mask, "ANALYSED_DATE"].min()

            summary_rows.append({
                "ANALYTE_CODE": analyte,
                "drift_detected": drift_detected,
                "drift_direction": drift_direction,
                "drift_start": drift_start,
                "severity_score": round(severity_score, 4),
                "first_warning_date": first_warning_date,
                "first_failure_date": first_failure_date,
                "n_observations": len(adf),
            })
            analyte_data[analyte] = adf

        summary = (
            pd.DataFrame(summary_rows)
            .sort_values(["drift_detected", "severity_score"], ascending=[False, False])
            .reset_index(drop=True)
        )
        return summary, analyte_data
