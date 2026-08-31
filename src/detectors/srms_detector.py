from __future__ import annotations

import argparse
import csv
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_COLUMNS = ["SRMSStandardCode", "SchemeCode", "Analyte"]
MODEL_VERSION = "srms-iforest-v0.3"
RULE_VERSION = "srms-qc-rules-v0.3"
IFOREST_TRAINING_MODE = "batch_retrospective"
IFOREST_FEATURE_COLUMNS = [
    "DeviationPct",
    "RobustZScore",
    "WithinRangePct",
    "RollingSlopePct",
    "RollingMeanShiftPct",
    "RollingMeanDiffPctTarget",
    "DiffFromPreviousPctTarget",
    "DistanceToLowerPctSpan",
    "DistanceToUpperPctSpan",
    "RollingStdPctTarget",
]
CCLAS_IFOREST_FEATURE_COLUMNS = [
    "DeviationPct",
    "RobustZScore",
    "RollingSlopePctTarget",
    "RollingStdPctTarget",
    "WithinRangePct",
    "DistanceToLowerPctSpan",
    "DistanceToUpperPctSpan",
]
CCLAS_TSV_PROPERTY_MAP = {
    "S_JobCode": "Job",
    "S_JobName": "JobName",
    "S_SampleCode": "SampleCode",
    "S_SampleName": "SampleName",
    "S_ClientSampleName": "ClientSampleName",
    "S_PrimaryAnalyticalType": "PrimaryType",
    "S_QcTypeCode": "QCType",
    "S_StandardCode": "StandardCode",
    "S_StandardLotCode": "StandardLotCode",
    "RSC_SchemeCode": "Scheme",
    "RSC_SchemeName": "SchemeName",
    "RSA_AnalyteCode": "Analyte",
    "RSA_AnalyteName": "AnalyteName",
    "SSA_UnitCode": "Unit",
    "SSA_NumericFinalValue": "Result",
    "PA_InternalMaxInclusive": "UpperLimitInclusive",
    "PA_InternalMaxValue": "UpperLimit",
    "PA_InternalMaxWarningInclusive": "UpperWarningInclusive",
    "PA_InternalMaxWarningValue": "UpperWarning",
    "PA_InternalMinInclusive": "LowerLimitInclusive",
    "PA_InternalMinValue": "LowerLimit",
    "PA_InternalMinWarningInclusive": "LowerWarningInclusive",
    "PA_InternalMinWarningValue": "LowerWarning",
    "PA_InternalTargetValue": "Target",
    "SSA_LimitStatus": "CCLAS_LimitStatus",
    "SSA_StandardStatus": "CCLAS_StandardStatus",
    "SSA_PrecisionStatus": "CCLAS_PrecisionStatus",
    "SSA_SpecificationStatus": "CCLAS_SpecificationStatus",
    "SSA_WorkflowStatus": "CCLAS_WorkflowStatus",
    "SSA_AnalysedByUserCode": "AnalysedByUserCode",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "srms_detector.yaml"


@dataclass(frozen=True)
class SRMSConfig:
    warning_margin_pct: float = 0.10
    robust_z_threshold: float = 3.5
    strong_robust_z_threshold: float = 4.5
    historical_sheet: str | None = None
    rolling_window: int = 5
    min_history: int = 5
    min_iforest_history: int = 20
    min_drift_history: int = 5
    drift_slope_pct_threshold: float = 0.5
    drift_mean_shift_pct_threshold: float = 1.0
    drift_consecutive_points: int = 4
    contamination: float = 0.05
    random_state: int = 42
    n_estimators: int = 250


REQUIRED_COLUMNS = [
    "ANALYTICAL_TYPE",
    "STD_LOT_CODE",
    "STD_CODE",
    "JOB_CODE",
    "NUMERIC_FINAL_VALUE",
    "ANALYSED_DATE",
    "SCHEME_CODE",
    "ANALYTE_CODE",
    "INTERNAL_MIN_VALUE",
    "INTERNAL_MAX_VALUE",
    "INTERNAL_MIN_INCLUSIVE",
    "INTERNAL_MAX_INCLUSIVE",
    "INTERNAL_MAX_WARNING_VALUE",
    "INTERNAL_MIN_WARNING_VALUE",
    "INTERNAL_MIN_WARNING_INCLUSIVE",
    "INTERNAL_MAX_WARNING_INCLUSIVE",
    "INTERNAL_TARGET_VALUE",
    "PARENT_NUMERIC_FINAL_VALUE",
    "UNIT_CODE",
    "SPECIFICATION_CODE",
]


OUTPUT_COLUMNS = [
    "SRMSStandardCode",
    "SchemeCode",
    "Analyte",
    "Value",
    "AnalysisDate",
    "TargetValue",
    "LowerLimit",
    "UpperLimit",
    "Deviation",
    "DeviationPct",
    "HistoricalMean",
    "HistoricalMedian",
    "HistoricalStd",
    "HistoricalMAD",
    "RobustZScore",
    "RollingMean",
    "RollingStd",
    "RollingMeanDiff",
    "PreviousValue",
    "DiffFromPrevious",
    "RollingSlope",
    "ConsecutiveIncreases",
    "ConsecutiveDecreases",
    "RollingSlopePct",
    "RollingMeanShiftPct",
    "RollingMeanDiffPctTarget",
    "DiffFromPreviousPctTarget",
    "DistanceToLowerPctSpan",
    "DistanceToUpperPctSpan",
    "RollingStdPctTarget",
    "IForestScore",
    "IForestScoreNorm",
    "IForestAnomaly",
    "IForestTrainingMode",
    "IForestFeatureList",
    "StatisticalAnomalyFlag",
    "DriftFlag",
    "LimitStatus",
    "FinalRiskLevel",
    "RiskReason",
]
CCLAS_STD_OUTPUT_COLUMNS = [
    "Job",
    "JobName",
    "SampleCode",
    "SampleName",
    "ClientSampleName",
    "PrimaryType",
    "QCType",
    "StandardCode",
    "StandardLotCode",
    "Scheme",
    "Analyte",
    "Unit",
    "Result",
    "Target",
    "LowerLimit",
    "LowerWarning",
    "UpperWarning",
    "UpperLimit",
    "Deviation",
    "DeviationPct",
    "HistorySource",
    "HistoryCount",
    "HistoricalMedian",
    "HistoricalMAD",
    "RobustZScore",
    "RollingMean",
    "RollingStd",
    "RollingSlope",
    "RollingSlopePctTarget",
    "StatisticalAnomalyFlag",
    "DriftFlag",
    "IsolationForestScore",
    "IsolationForestAnomaly",
    "IsolationForestStatus",
    "Limit_Status",
    "Historical_Status",
    "Final_Risk",
    "Detection_Method",
    "Reason",
    "CCLAS_StandardStatus",
    "CCLAS_StandardStatus_Normalized",
    "CCLAS_Mismatch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SRMS candidate anomaly detection pipeline.")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "ResultSet.csv"),
        help="Path to a CSV, Excel workbook, or zip containing a CSV.",
    )
    parser.add_argument("--sheet", default=None, help="Optional sheet name for Excel workbooks.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "srms_outputs"),
        help="Directory for SRMS outputs and charts.",
    )
    parser.add_argument(
        "--results-csv",
        default=str(PROJECT_ROOT / "srms_candidate_anomaly_results.csv"),
        help="Detailed SRMS result CSV path.",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(PROJECT_ROOT / "srms_candidate_summary.csv"),
        help="Risk summary CSV path.",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Optional historical CSV/Excel/zip source used to score current CCLAS TSV Standard rows.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to SRMS detector YAML configuration.",
    )
    parser.add_argument("--contamination", type=float, default=None)
    parser.add_argument("--warning-margin-pct", type=float, default=None)
    parser.add_argument("--robust-z-threshold", type=float, default=None)
    parser.add_argument("--strong-robust-z-threshold", type=float, default=None)
    parser.add_argument("--rolling-window", type=int, default=None)
    parser.add_argument("--min-history", type=int, default=None)
    parser.add_argument("--min-iforest-history", type=int, default=None)
    parser.add_argument("--min-drift-history", type=int, default=None)
    parser.add_argument("--drift-slope-pct-threshold", type=float, default=None)
    parser.add_argument("--drift-mean-shift-pct-threshold", type=float, default=None)
    parser.add_argument("--drift-consecutive-points", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


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


def _parse_simple_yaml(path: Path) -> dict[str, object]:
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]
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
            child: dict[str, object] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_config_scalar(value)
    return root


def load_config_file(path_str: str | None) -> dict[str, object]:
    if not path_str:
        return {}
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        return {}
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except ModuleNotFoundError:
        return _parse_simple_yaml(path)


def load_srms_config(path_str: str | None = None) -> SRMSConfig:
    data = load_config_file(path_str or str(DEFAULT_CONFIG_PATH))
    srms = data.get("srms", {}) if isinstance(data, dict) else {}
    if not isinstance(srms, dict):
        return SRMSConfig()

    historical = srms.get("historical", {})
    isolation_forest = srms.get("isolation_forest", {})
    drift = srms.get("drift", {})
    historical = historical if isinstance(historical, dict) else {}
    isolation_forest = isolation_forest if isinstance(isolation_forest, dict) else {}
    drift = drift if isinstance(drift, dict) else {}

    return SRMSConfig(
        warning_margin_pct=float(srms.get("warning_margin_pct", SRMSConfig.warning_margin_pct)),
        robust_z_threshold=float(srms.get("robust_z_threshold", SRMSConfig.robust_z_threshold)),
        strong_robust_z_threshold=float(srms.get("strong_robust_z_threshold", SRMSConfig.strong_robust_z_threshold)),
        historical_sheet=historical.get("sheet", SRMSConfig.historical_sheet),
        rolling_window=int(historical.get("rolling_window", SRMSConfig.rolling_window)),
        min_history=int(historical.get("min_history", SRMSConfig.min_history)),
        min_iforest_history=int(isolation_forest.get("min_history", SRMSConfig.min_iforest_history)),
        min_drift_history=int(drift.get("min_history", SRMSConfig.min_drift_history)),
        drift_slope_pct_threshold=float(drift.get("slope_pct_threshold", SRMSConfig.drift_slope_pct_threshold)),
        drift_mean_shift_pct_threshold=float(drift.get("mean_shift_pct_threshold", SRMSConfig.drift_mean_shift_pct_threshold)),
        drift_consecutive_points=int(drift.get("consecutive_points", SRMSConfig.drift_consecutive_points)),
        contamination=float(isolation_forest.get("contamination", SRMSConfig.contamination)),
        random_state=int(isolation_forest.get("random_state", SRMSConfig.random_state)),
        n_estimators=int(isolation_forest.get("n_estimators", SRMSConfig.n_estimators)),
    )


def config_from_args(args: argparse.Namespace) -> SRMSConfig:
    config = load_srms_config(args.config)
    values = {
        "warning_margin_pct": config.warning_margin_pct if args.warning_margin_pct is None else args.warning_margin_pct,
        "robust_z_threshold": config.robust_z_threshold if args.robust_z_threshold is None else args.robust_z_threshold,
        "strong_robust_z_threshold": (
            config.strong_robust_z_threshold if args.strong_robust_z_threshold is None else args.strong_robust_z_threshold
        ),
        "historical_sheet": config.historical_sheet,
        "rolling_window": config.rolling_window if args.rolling_window is None else args.rolling_window,
        "min_history": config.min_history if args.min_history is None else args.min_history,
        "min_iforest_history": config.min_iforest_history if args.min_iforest_history is None else args.min_iforest_history,
        "min_drift_history": config.min_drift_history if args.min_drift_history is None else args.min_drift_history,
        "drift_slope_pct_threshold": (
            config.drift_slope_pct_threshold if args.drift_slope_pct_threshold is None else args.drift_slope_pct_threshold
        ),
        "drift_mean_shift_pct_threshold": (
            config.drift_mean_shift_pct_threshold
            if args.drift_mean_shift_pct_threshold is None
            else args.drift_mean_shift_pct_threshold
        ),
        "drift_consecutive_points": (
            config.drift_consecutive_points if args.drift_consecutive_points is None else args.drift_consecutive_points
        ),
        "contamination": config.contamination if args.contamination is None else args.contamination,
        "random_state": config.random_state,
        "n_estimators": config.n_estimators,
    }
    values["contamination"] = min(max(values["contamination"], 0.001), 0.2)
    return SRMSConfig(**values)



def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.where(denominator.abs() > 1e-12)
    return (numerator / safe_denominator).replace([np.inf, -np.inf], np.nan)


def load_data(path_str: str, sheet_name: str | None = None, max_rows: int | None = None) -> pd.DataFrame:
    path = Path(path_str).expanduser().resolve()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False, nrows=max_rows)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name, nrows=max_rows)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                raise ValueError(f"No CSV file found inside {path}")
            with archive.open(csv_members[0]) as handle:
                return pd.read_csv(handle, low_memory=False, nrows=max_rows)
    raise ValueError(f"Unsupported input type: {path.suffix}")


def is_cclas_object_property_tsv(path_str: str) -> bool:
    path = Path(path_str).expanduser()
    if path.suffix.lower() != ".tsv":
        return False
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, [])
    return [column.strip().upper() for column in header[:3]] == ["OBJECT", "PROPERTY", "VALUE"]


def normalize_cclas_qc_type(primary_type: object, qc_type: object) -> str:
    raw = str(qc_type or "").strip().upper()
    primary = str(primary_type or "").strip().upper()
    if raw:
        return raw
    return "STD" if primary == "STANDARD" else primary


def parse_cclas_tsv(path_str: str) -> pd.DataFrame:
    path = Path(path_str).expanduser().resolve()
    records: list[dict[str, object]] = []
    current_sample: dict[str, object] | None = None
    current_result: dict[str, object] = {}
    sample_block = 0

    def flush_result() -> None:
        if current_sample is None or not current_result:
            return
        record = {**current_sample, **current_result}
        record["ResultRow"] = len(records) + 1
        records.append(record)

    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"OBJECT", "PROPERTY", "VALUE"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must contain OBJECT, PROPERTY, VALUE columns")

        for row in reader:
            obj = str(row.get("OBJECT", "")).strip().rstrip(":")
            prop = str(row.get("PROPERTY", "")).strip().rstrip("=")
            value = row.get("VALUE", "")
            mapped = CCLAS_TSV_PROPERTY_MAP.get(prop, prop)

            if obj == "VSAMPLE" and prop == "S_JobCode":
                flush_result()
                sample_block += 1
                current_sample = {"SampleBlock": sample_block, mapped: value}
                current_result = {}
                continue

            if obj == "VSAMPLE":
                if current_sample is None:
                    sample_block += 1
                    current_sample = {"SampleBlock": sample_block}
                current_sample[mapped] = value
                continue

            if obj == "VSSA":
                if prop == "RSC_SchemeCode" and current_result:
                    flush_result()
                    current_result = {}
                current_result[mapped] = value

    flush_result()
    parsed = pd.DataFrame(records)
    if parsed.empty:
        return parsed

    parsed["QCType"] = [
        normalize_cclas_qc_type(primary, qc)
        for primary, qc in zip(parsed.get("PrimaryType"), parsed.get("QCType"))
    ]
    parsed["PrimaryType"] = parsed["PrimaryType"].replace("", np.nan).fillna(parsed["QCType"])
    numeric_columns = ["Result", "Target", "LowerLimit", "LowerWarning", "UpperWarning", "UpperLimit"]
    for column in numeric_columns:
        if column in parsed.columns:
            parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    return parsed


def extract_cclas_standard_records(parsed: pd.DataFrame) -> pd.DataFrame:
    if parsed.empty:
        return parsed.copy()
    standard_mask = parsed["QCType"].eq("STD") | parsed["PrimaryType"].astype(str).str.upper().eq("STANDARD")
    standards = parsed.loc[standard_mask].copy()
    standards["ReferenceMaterialCandidate"] = True
    standards["CandidateIdentificationNote"] = (
        "Standard/SRMS candidate selected from CCLAS TSV where S_QcTypeCode is STD "
        "or S_PrimaryAnalyticalType is STANDARD."
    )
    return standards.reset_index(drop=True)


def format_qc_number(value: object) -> str:
    if pd.isna(value):
        return "missing"
    return f"{float(value):.6g}"


def detect_standard_limit_status(row: pd.Series) -> tuple[str, str]:
    result = row.get("Result")
    lower = row.get("LowerLimit")
    upper = row.get("UpperLimit")
    lower_warning = row.get("LowerWarning")
    upper_warning = row.get("UpperWarning")
    lower_inclusive = str(row.get("LowerLimitInclusive", "Y") or "Y").strip().upper() != "N"
    upper_inclusive = str(row.get("UpperLimitInclusive", "Y") or "Y").strip().upper() != "N"
    lower_warning_inclusive = str(row.get("LowerWarningInclusive", "Y") or "Y").strip().upper() != "N"
    upper_warning_inclusive = str(row.get("UpperWarningInclusive", "Y") or "Y").strip().upper() != "N"

    if pd.isna(result):
        return "NOT_EVALUATED", "Numeric result is missing."
    if pd.isna(lower) or pd.isna(upper):
        return "NOT_EVALUATED", "Lower or upper acceptance limit is missing."

    lower_fail = result < lower if lower_inclusive else result <= lower
    upper_fail = result > upper if upper_inclusive else result >= upper
    lower_warning_hit = (
        pd.notna(lower_warning)
        and (result < lower_warning if lower_warning_inclusive else result <= lower_warning)
    )
    upper_warning_hit = (
        pd.notna(upper_warning)
        and (result > upper_warning if upper_warning_inclusive else result >= upper_warning)
    )

    if lower_fail:
        return "FAIL", f"Result {format_qc_number(result)} is below the lower acceptance limit {format_qc_number(lower)}."
    if upper_fail:
        return "FAIL", f"Result {format_qc_number(result)} is above the upper acceptance limit {format_qc_number(upper)}."
    if lower_warning_hit:
        return (
            "WARNING",
            f"Result {format_qc_number(result)} is inside acceptance limits but below lower warning limit {format_qc_number(lower_warning)}.",
        )
    if upper_warning_hit:
        return (
            "WARNING",
            f"Result {format_qc_number(result)} is inside acceptance limits but above upper warning limit {format_qc_number(upper_warning)}.",
        )
    return "PASS", "Result is within the configured acceptance and warning limits."


def add_cclas_standard_historical_features(standards: pd.DataFrame, config: SRMSConfig) -> pd.DataFrame:
    working = standards.sort_values(["StandardCode", "Scheme", "Analyte", "ResultRow"], na_position="last").copy()
    grouped = working.groupby(["StandardCode", "Scheme", "Analyte"], dropna=False, sort=False)
    working["HistoryCount"] = grouped.cumcount()
    working["Historical_Median"] = grouped["Result"].transform(lambda s: s.expanding().median().shift(1))
    working["MAD"] = grouped["Result"].transform(lambda s: s.expanding().apply(_mad, raw=False).shift(1))
    shifted = grouped["Result"].shift(1).groupby(
        [working["StandardCode"], working["Scheme"], working["Analyte"]],
        dropna=False,
        sort=False,
    )
    working["Rolling_Mean"] = shifted.transform(lambda s: s.rolling(config.rolling_window, min_periods=2).mean())
    working["Rolling_Std"] = shifted.transform(lambda s: s.rolling(config.rolling_window, min_periods=2).std())
    working["Rolling_Slope"] = shifted.transform(lambda s: s.rolling(config.rolling_window, min_periods=3).apply(_rolling_slope, raw=False))
    mad_scale = working["MAD"].replace(0, np.nan)
    working["Robust_Z"] = 0.6745 * (working["Result"] - working["Historical_Median"]) / mad_scale
    target_abs = working["Target"].abs()
    working["Rolling_Std_Pct_Target"] = safe_divide(working["Rolling_Std"], target_abs) * 100
    working["Rolling_Slope_Pct_Target"] = safe_divide(working["Rolling_Slope"], target_abs) * 100
    working.loc[working["HistoryCount"] < config.min_history, "Robust_Z"] = np.nan
    working["Historical_Status"] = "INSUFFICIENT_HISTORY"
    has_history = working["HistoryCount"].ge(config.min_history)
    working.loc[has_history, "Historical_Status"] = "PASS"
    working.loc[has_history & working["Robust_Z"].abs().ge(config.robust_z_threshold), "Historical_Status"] = "WARNING"
    return working.sort_values("ResultRow").reset_index(drop=True)


def add_cclas_standard_isolation_forest(standards: pd.DataFrame, config: SRMSConfig) -> pd.DataFrame:
    working = standards.copy()
    eligible = working["HistoryCount"].ge(config.min_iforest_history)
    feature_columns = [column for column in CCLAS_IFOREST_FEATURE_COLUMNS if column in working.columns]
    feature_frame = working[feature_columns].replace([np.inf, -np.inf], np.nan) if feature_columns else pd.DataFrame(index=working.index)
    min_training_rows = max(config.min_iforest_history, 10)

    working["IsolationForest_Score"] = np.nan
    working["IsolationForest_Anomaly"] = False
    working["IsolationForest_Status"] = "INSUFFICIENT_HISTORY"
    working.loc[eligible, "IsolationForest_Status"] = "INSUFFICIENT_MODEL_HISTORY"

    if not eligible.any() or not feature_columns:
        return working

    from sklearn.ensemble import IsolationForest

    for idx in working.index[eligible]:
        current_row_number = working.at[idx, "ResultRow"]
        train_mask = working["ResultRow"].lt(current_row_number) & working["HistoryCount"].ge(config.min_history)
        train_frame = feature_frame.loc[train_mask]
        usable_features = [column for column in feature_columns if train_frame[column].notna().any()]
        if len(train_frame) < min_training_rows or not usable_features:
            continue

        train_frame = train_frame[usable_features].copy()
        medians = train_frame.median()
        train_frame = train_frame.fillna(medians).fillna(0)
        current_frame = feature_frame.loc[[idx], usable_features].fillna(medians).fillna(0)

        model = IsolationForest(
            n_estimators=config.n_estimators,
            contamination=config.contamination,
            random_state=config.random_state,
            n_jobs=-1,
        )
        model.fit(train_frame)
        working.at[idx, "IsolationForest_Anomaly"] = bool(model.predict(current_frame)[0] == -1)
        working.at[idx, "IsolationForest_Score"] = float(-model.score_samples(current_frame)[0])
        working.at[idx, "IsolationForest_Status"] = "WARNING" if working.at[idx, "IsolationForest_Anomaly"] else "PASS"
    return working


def assign_cclas_standard_final_risk(row: pd.Series) -> str:
    if row["Limit_Status"] == "FAIL":
        return "Critical"
    supporting_anomaly = (
        row["Historical_Status"] == "WARNING"
        or bool(row.get("IsolationForestAnomaly", row.get("IsolationForest_Anomaly", False)))
        or bool(row.get("DriftFlag", False))
    )
    if row["Limit_Status"] == "WARNING":
        return "High" if supporting_anomaly else "Medium"
    if row["Limit_Status"] == "PASS" and supporting_anomaly:
        return "Medium"
    if row["Limit_Status"] == "PASS":
        return "Low"
    return "Not_Evaluated"


def assign_cclas_standard_detection_method(row: pd.Series) -> str:
    if row["Limit_Status"] in {"FAIL", "WARNING", "PASS"}:
        method = "Known QC limit rule"
    else:
        method = "Not evaluated"
    supporting = []
    if row.get("Historical_Status") == "WARNING":
        supporting.append("historical robust z-score")
    if bool(row.get("IsolationForestAnomaly", row.get("IsolationForest_Anomaly", False))):
        supporting.append("point-in-time Isolation Forest")
    if bool(row.get("DriftFlag", False)):
        supporting.append("historical drift")
    if supporting:
        method = f"{method} with {' and '.join(supporting)} support"
    return method


def normalize_cclas_standard_status(status: object) -> str:
    text = str(status or "").strip().upper()
    if text in {"", "NAN", "NOT_REQUIRED", "NOT_APPLICABLE"}:
        return "NOT_EVALUATED"
    if "FAIL" in text:
        return "FAIL"
    if "WARN" in text:
        return "WARNING"
    if text == "PASS":
        return "PASS"
    return text


def prepare_historical_srms_baseline(
    history_path: str | None,
    config: SRMSConfig,
    sheet_name: str | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    if not history_path:
        return pd.DataFrame()
    raw_history = load_data(history_path, sheet_name=sheet_name, max_rows=max_rows)
    historical = extract_srms_candidates(raw_history)
    historical = apply_known_source_corrections(historical)
    historical = add_known_limit_features(historical, config)
    historical = add_historical_features(historical, config)
    return historical.sort_values(GROUP_COLUMNS + ["AnalysisDate", "SampleID"], na_position="last").reset_index(drop=True)


def matching_historical_group(historical: pd.DataFrame, current_row: pd.Series) -> pd.DataFrame:
    if historical.empty:
        return historical.copy()
    mask = (
        historical["SRMSStandardCode"].eq(current_row.get("StandardCode"))
        & historical["SchemeCode"].eq(current_row.get("Scheme"))
        & historical["Analyte"].eq(current_row.get("Analyte"))
    )
    group = historical.loc[mask].copy()
    current_date = pd.to_datetime(current_row.get("AnalysisDate"), errors="coerce")
    if pd.notna(current_date) and "AnalysisDate" in group.columns:
        group = group[group["AnalysisDate"].lt(current_date)]
    return group.sort_values(["AnalysisDate", "SampleID"], na_position="last")


def add_cclas_standard_current_features(
    standards: pd.DataFrame,
    historical: pd.DataFrame,
    config: SRMSConfig,
    history_source: str | None,
) -> pd.DataFrame:
    working = standards.copy()
    defaults: dict[str, object] = {
        "HistorySource": "not supplied" if not history_source else str(Path(history_source).name),
        "HistoryCount": 0,
        "HistoricalMedian": np.nan,
        "HistoricalMAD": np.nan,
        "RobustZScore": np.nan,
        "RollingMean": np.nan,
        "RollingStd": np.nan,
        "RollingSlope": np.nan,
        "RollingSlopePctTarget": np.nan,
        "RollingStdPctTarget": np.nan,
        "StatisticalAnomalyFlag": False,
        "Historical_Status": "INSUFFICIENT_HISTORY",
        "DriftFlag": False,
    }
    for column, value in defaults.items():
        working[column] = value

    for idx, row in working.iterrows():
        group = matching_historical_group(historical, row)
        values = group["Value"].dropna() if "Value" in group.columns else pd.Series(dtype=float)
        count = int(len(values))
        working.at[idx, "HistoryCount"] = count

        if count < config.min_history:
            continue

        median = float(values.median())
        mad = _mad(values)
        latest = values.tail(config.rolling_window)
        rolling_mean = float(latest.mean()) if len(latest) >= 2 else np.nan
        rolling_std = float(latest.std()) if len(latest) >= 2 else np.nan
        rolling_slope = _rolling_slope(latest) if len(latest) >= 3 else np.nan

        result = row.get("Result")
        target = row.get("Target")
        target_abs = abs(target) if pd.notna(target) else np.nan
        robust_z = np.nan
        if pd.notna(result) and pd.notna(mad) and abs(mad) > 1e-12:
            robust_z = 0.6745 * (result - median) / mad
        elif pd.notna(result) and pd.notna(mad) and abs(mad) <= 1e-12 and not np.isclose(result, median):
            robust_z = np.inf if result > median else -np.inf

        rolling_slope_pct = rolling_slope / target_abs * 100 if pd.notna(rolling_slope) and target_abs > 1e-12 else np.nan
        rolling_std_pct = rolling_std / target_abs * 100 if pd.notna(rolling_std) and target_abs > 1e-12 else np.nan
        statistical_flag = bool(pd.notna(robust_z) and abs(robust_z) >= config.robust_z_threshold)

        working.at[idx, "HistoricalMedian"] = median
        working.at[idx, "HistoricalMAD"] = mad
        working.at[idx, "RobustZScore"] = robust_z
        working.at[idx, "RollingMean"] = rolling_mean
        working.at[idx, "RollingStd"] = rolling_std
        working.at[idx, "RollingSlope"] = rolling_slope
        working.at[idx, "RollingSlopePctTarget"] = rolling_slope_pct
        working.at[idx, "RollingStdPctTarget"] = rolling_std_pct
        working.at[idx, "StatisticalAnomalyFlag"] = statistical_flag
        working.at[idx, "Historical_Status"] = "WARNING" if statistical_flag else "PASS"

        if count >= config.min_drift_history and pd.notna(result) and pd.notna(target) and target_abs > 1e-12:
            recent_mean_shift_pct = (rolling_mean - target) / target_abs * 100 if pd.notna(rolling_mean) else np.nan
            last_value = values.iloc[-1] if len(values) else np.nan
            current_deviation = result - target
            rolling_deviation = rolling_mean - target if pd.notna(rolling_mean) else np.nan
            moving_same_side = pd.notna(rolling_deviation) and np.sign(current_deviation) == np.sign(rolling_deviation)
            further_from_target = pd.notna(rolling_deviation) and abs(current_deviation) > abs(rolling_deviation)
            continuing_slope = (
                pd.notna(rolling_slope)
                and pd.notna(last_value)
                and ((rolling_slope > 0 and result > last_value) or (rolling_slope < 0 and result < last_value))
            )
            strong_slope = pd.notna(rolling_slope_pct) and abs(rolling_slope_pct) >= config.drift_slope_pct_threshold
            mean_shift = pd.notna(recent_mean_shift_pct) and abs(recent_mean_shift_pct) >= config.drift_mean_shift_pct_threshold
            working.at[idx, "DriftFlag"] = bool(moving_same_side and further_from_target and continuing_slope and (strong_slope or mean_shift))

    alias_map = {
        "HistoricalMedian": "Historical_Median",
        "HistoricalMAD": "MAD",
        "RobustZScore": "Robust_Z",
        "RollingMean": "Rolling_Mean",
        "RollingStd": "Rolling_Std",
        "RollingSlope": "Rolling_Slope",
    }
    for source, alias in alias_map.items():
        working[alias] = working[source]
    working["Rolling_Slope_Pct_Target"] = working["RollingSlopePctTarget"]
    working["Rolling_Std_Pct_Target"] = working["RollingStdPctTarget"]
    return working


def add_cclas_standard_historical_isolation_forest(
    standards: pd.DataFrame,
    historical: pd.DataFrame,
    config: SRMSConfig,
) -> pd.DataFrame:
    working = standards.copy()
    working["IsolationForestScore"] = np.nan
    working["IsolationForestAnomaly"] = False
    working["IsolationForestStatus"] = "INSUFFICIENT_HISTORY"

    if historical.empty:
        working["IsolationForest_Score"] = working["IsolationForestScore"]
        working["IsolationForest_Anomaly"] = working["IsolationForestAnomaly"]
        working["IsolationForest_Status"] = working["IsolationForestStatus"]
        return working

    train_feature_map = {
        "DeviationPct": "DeviationPct",
        "RobustZScore": "RobustZScore",
        "RollingSlopePctTarget": "RollingSlopePct",
        "RollingStdPctTarget": "RollingStdPctTarget",
        "WithinRangePct": "WithinRangePct",
        "DistanceToLowerPctSpan": "DistanceToLowerPctSpan",
        "DistanceToUpperPctSpan": "DistanceToUpperPctSpan",
    }
    train = historical.copy()
    for current_name, history_name in train_feature_map.items():
        if history_name in train.columns:
            train[current_name] = train[history_name]

    train_columns = [column for column in CCLAS_IFOREST_FEATURE_COLUMNS if column in train.columns and column in working.columns]
    eligible_train = train["HistoryCount"].ge(config.min_iforest_history) if "HistoryCount" in train.columns else pd.Series(False, index=train.index)
    eligible_current = working["HistoryCount"].ge(config.min_iforest_history)
    working.loc[eligible_current, "IsolationForestStatus"] = "INSUFFICIENT_MODEL_HISTORY"

    if not train_columns:
        working["IsolationForest_Score"] = working["IsolationForestScore"]
        working["IsolationForest_Anomaly"] = working["IsolationForestAnomaly"]
        working["IsolationForest_Status"] = working["IsolationForestStatus"]
        return working

    train_frame = train.loc[eligible_train, train_columns].replace([np.inf, -np.inf], np.nan)
    usable_features = [column for column in train_columns if train_frame[column].notna().any()]
    if len(train_frame) < max(config.min_iforest_history, 10) or not usable_features:
        working["IsolationForest_Score"] = working["IsolationForestScore"]
        working["IsolationForest_Anomaly"] = working["IsolationForestAnomaly"]
        working["IsolationForest_Status"] = working["IsolationForestStatus"]
        return working

    from sklearn.ensemble import IsolationForest

    train_frame = train_frame[usable_features]
    medians = train_frame.median()
    train_frame = train_frame.fillna(medians).fillna(0)
    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(train_frame)

    current_idx = working.index[eligible_current]
    if len(current_idx):
        current_frame = working.loc[current_idx, usable_features].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
        predictions = model.predict(current_frame)
        scores = -model.score_samples(current_frame)
        working.loc[current_idx, "IsolationForestAnomaly"] = predictions == -1
        working.loc[current_idx, "IsolationForestScore"] = scores
        working.loc[current_idx, "IsolationForestStatus"] = np.where(predictions == -1, "WARNING", "PASS")

    working["IsolationForest_Score"] = working["IsolationForestScore"]
    working["IsolationForest_Anomaly"] = working["IsolationForestAnomaly"]
    working["IsolationForest_Status"] = working["IsolationForestStatus"]
    return working


def build_cclas_standard_reason(row: pd.Series) -> str:
    base_reason = str(row.get("Reason") or "")
    if row.get("Limit_Status") == "FAIL":
        return base_reason

    additions = []
    if row.get("Historical_Status") == "INSUFFICIENT_HISTORY":
        additions.append(
            f"historical assessment is unavailable because only {int(row.get('HistoryCount', 0))} matching historical observations were found"
        )
    elif row.get("StatisticalAnomalyFlag"):
        additions.append(f"unusual compared with the historical SRMS baseline (Robust Z = {format_qc_number(row.get('RobustZScore'))})")
    elif row.get("Historical_Status") == "PASS":
        additions.append(f"historical baseline check passed using {int(row.get('HistoryCount', 0))} matching observations")

    if bool(row.get("IsolationForestAnomaly", False)):
        additions.append("Isolation Forest marked the group-relative feature pattern as unusual")
    elif row.get("IsolationForestStatus") == "INSUFFICIENT_HISTORY":
        additions.append("Isolation Forest was not run because historical training data was insufficient")

    if bool(row.get("DriftFlag", False)):
        additions.append("recent historical trend and the current result suggest drift away from target")

    if not additions:
        return base_reason
    return f"{base_reason} {'; '.join(additions)}."


def run_cclas_standard_pipeline(
    path_str: str,
    config: SRMSConfig,
    history_path: str | None = None,
    sheet_name: str | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parsed = parse_cclas_tsv(path_str)
    standards = extract_cclas_standard_records(parsed)
    if standards.empty:
        status_summary = pd.DataFrame(columns=["Limit_Status", "Count"])
        risk_summary = pd.DataFrame(columns=["Final_Risk", "Count"])
        return standards, status_summary, risk_summary

    if "CCLAS_StandardStatus" not in standards.columns:
        standards["CCLAS_StandardStatus"] = np.nan

    historical = prepare_historical_srms_baseline(history_path, config, sheet_name=sheet_name, max_rows=max_rows)

    limit_results = standards.apply(detect_standard_limit_status, axis=1, result_type="expand")
    standards["Limit_Status"] = limit_results[0]
    standards["Reason"] = limit_results[1]
    standards["Deviation"] = standards["Result"] - standards["Target"]
    standards["DeviationPct"] = safe_divide(standards["Deviation"], standards["Target"].abs()) * 100
    span = standards["UpperLimit"] - standards["LowerLimit"]
    standards["WithinRangePct"] = safe_divide(standards["Result"] - standards["LowerLimit"], span) * 100
    standards["DistanceToLowerPctSpan"] = safe_divide(standards["Result"] - standards["LowerLimit"], span) * 100
    standards["DistanceToUpperPctSpan"] = safe_divide(standards["UpperLimit"] - standards["Result"], span) * 100
    standards = add_cclas_standard_current_features(standards, historical, config, history_path)
    standards = add_cclas_standard_historical_isolation_forest(standards, historical, config)
    standards["ML_Anomaly"] = standards["IsolationForestAnomaly"]
    standards["Final_Risk"] = standards.apply(assign_cclas_standard_final_risk, axis=1)
    standards["Detection_Method"] = standards.apply(assign_cclas_standard_detection_method, axis=1)
    standards["Reason"] = standards.apply(build_cclas_standard_reason, axis=1)
    standards["CCLAS_StandardStatus_Normalized"] = standards["CCLAS_StandardStatus"].apply(normalize_cclas_standard_status)
    standards["CCLAS_Mismatch"] = standards["Limit_Status"].apply(normalize_cclas_standard_status).ne(
        standards["CCLAS_StandardStatus_Normalized"]
    )
    standards.loc[
        standards["Limit_Status"].eq("NOT_EVALUATED") & standards["CCLAS_StandardStatus_Normalized"].eq("NOT_EVALUATED"),
        "CCLAS_Mismatch",
    ] = False

    status_summary = standards["Limit_Status"].value_counts(dropna=False).rename_axis("Limit_Status").reset_index(name="Count")
    risk_summary = standards["Final_Risk"].value_counts(dropna=False).rename_axis("Final_Risk").reset_index(name="Count")
    return standards, status_summary, risk_summary


def create_cclas_standard_validation_report(
    standards: pd.DataFrame,
    status_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
    output_dir: Path,
    input_path: str,
) -> Path:
    report_path = output_dir / "cclas_standard_validation_report.md"
    mismatches = standards[standards["CCLAS_Mismatch"]]
    lines = [
        "# CCLAS TSV Standard/SRMS Detection Validation",
        "",
        f"- Input file: `{input_path}`",
        f"- STD records evaluated: `{len(standards)}`",
        f"- PASS: `{int(standards['Limit_Status'].eq('PASS').sum())}`",
        f"- WARNING: `{int(standards['Limit_Status'].eq('WARNING').sum())}`",
        f"- FAIL: `{int(standards['Limit_Status'].eq('FAIL').sum())}`",
        f"- NOT_EVALUATED: `{int(standards['Limit_Status'].eq('NOT_EVALUATED').sum())}`",
        f"- Sufficient historical baseline: `{int(standards['Historical_Status'].ne('INSUFFICIENT_HISTORY').sum())}`",
        f"- Insufficient historical baseline: `{int(standards['Historical_Status'].eq('INSUFFICIENT_HISTORY').sum())}`",
        f"- Isolation Forest scored: `{int(standards['IsolationForestStatus'].isin(['PASS', 'WARNING']).sum())}`",
        f"- Isolation Forest anomalies: `{int(standards['IsolationForestAnomaly'].sum())}`",
        f"- Drift flags: `{int(standards['DriftFlag'].sum())}`",
        f"- CCLAS mismatches: `{len(mismatches)}`",
        "",
        "## Limit Status Summary",
        "",
        status_summary.to_markdown(index=False),
        "",
        "## Final Risk Summary",
        "",
        risk_summary.to_markdown(index=False),
        "",
        "## Scope Notes",
        "",
        "- Only `STD` / Standard analytical records are evaluated by this component.",
        "- `SSA_StandardStatus` is retained only for validation and is not used as a detection input.",
        "- Historical and Isolation Forest signals are supporting evidence only; known acceptance-limit failure always gives `Critical` final risk.",
        "- If the current TSV has no reliable `AnalysisDate`, the supplied historical source is assumed to contain historical-only observations; current TSV rows are never included in baseline statistics or model training.",
    ]
    if not mismatches.empty:
        lines.extend(
            [
                "",
                "## Mismatches for Investigation",
                "",
                mismatches[
                    [
                        "SampleCode",
                        "StandardCode",
                        "Scheme",
                        "Analyte",
                        "Result",
                        "Target",
                        "LowerLimit",
                        "LowerWarning",
                        "UpperWarning",
                        "UpperLimit",
                        "Limit_Status",
                        "CCLAS_StandardStatus",
                        "Reason",
                    ]
                ].to_markdown(index=False),
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def order_cclas_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    internal_aliases = {
        "Historical_Median",
        "MAD",
        "Robust_Z",
        "Rolling_Mean",
        "Rolling_Std",
        "Rolling_Slope",
        "Rolling_Slope_Pct_Target",
        "Rolling_Std_Pct_Target",
        "IsolationForest_Score",
        "IsolationForest_Anomaly",
        "IsolationForest_Status",
        "ML_Anomaly",
    }
    leading = [column for column in CCLAS_STD_OUTPUT_COLUMNS if column in df.columns]
    trailing = [column for column in df.columns if column not in leading and column not in internal_aliases]
    return df[leading + trailing]


def require_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def extract_srms_candidates(raw_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(raw_df)
    working = raw_df.copy()
    numeric_columns = [
        "NUMERIC_FINAL_VALUE",
        "INTERNAL_MIN_VALUE",
        "INTERNAL_MAX_VALUE",
        "INTERNAL_MIN_WARNING_VALUE",
        "INTERNAL_MAX_WARNING_VALUE",
        "INTERNAL_TARGET_VALUE",
        "PARENT_NUMERIC_FINAL_VALUE",
    ]
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["ANALYSED_DATE"] = pd.to_datetime(working["ANALYSED_DATE"], errors="coerce")

    standard_mask = working["ANALYTICAL_TYPE"].eq("Standard")
    reference_mask = ~working["STD_CODE"].fillna("").isin(["", "Sample", "TSV_BLANK"])
    value_mask = working["NUMERIC_FINAL_VALUE"].notna()
    target_mask = working["INTERNAL_TARGET_VALUE"].notna()
    limit_mask = working["INTERNAL_MIN_VALUE"].notna() & working["INTERNAL_MAX_VALUE"].notna()

    extracted = working.loc[standard_mask & reference_mask & value_mask & target_mask & limit_mask].copy()
    extracted = extracted.rename(
        columns={
            "STD_LOT_CODE": "SRMSLotCode",
            "STD_CODE": "SRMSStandardCode",
            "JOB_CODE": "JobCode",
            "NUMERIC_FINAL_VALUE": "Value",
            "ANALYSED_DATE": "AnalysisDate",
            "SCHEME_CODE": "SchemeCode",
            "ANALYTE_CODE": "Analyte",
            "STANDARD_STATUS": "StandardStatus",
            "INTERNAL_MIN_VALUE": "LowerLimit",
            "INTERNAL_MAX_VALUE": "UpperLimit",
            "INTERNAL_MIN_INCLUSIVE": "LowerLimitInclusive",
            "INTERNAL_MAX_INCLUSIVE": "UpperLimitInclusive",
            "INTERNAL_MIN_WARNING_VALUE": "WarningLower",
            "INTERNAL_MAX_WARNING_VALUE": "WarningUpper",
            "INTERNAL_MIN_WARNING_INCLUSIVE": "WarningLowerInclusive",
            "INTERNAL_MAX_WARNING_INCLUSIVE": "WarningUpperInclusive",
            "INTERNAL_TARGET_VALUE": "TargetValue",
            "PARENT_NUMERIC_FINAL_VALUE": "ParentValue",
            "UNIT_CODE": "UnitCode",
            "SPECIFICATION_CODE": "SpecificationCode",
        }
    )
    extracted["SampleID"] = extracted["JobCode"].fillna("JOB-UNKNOWN") + "-" + extracted.index.astype(str)
    extracted["SourceCategory"] = "Standard"
    extracted["ReferenceMaterialCandidate"] = True
    extracted["ReferenceCodeMatchesSpecification"] = extracted["SRMSStandardCode"].fillna("").eq(
        extracted["SpecificationCode"].fillna("")
    )
    extracted["CandidateIdentificationNote"] = (
        "SRMS/reference-material standard candidate selected from ANALYTICAL_TYPE=Standard "
        "with valid reference code, numeric result, target, and acceptance limits; no explicit SRMS subtype exists."
    )
    extracted["SRMSAssumption"] = (
        "Reference-material standard used as SRMS candidate because explicit SRMS subtype "
        "is not present in ResultSet.csv."
    )
    extracted["ModelVersion"] = MODEL_VERSION
    extracted["RuleVersion"] = RULE_VERSION
    extracted = extracted.drop(columns=["PRECISION_STATUS"], errors="ignore")
    return extracted.sort_values(["AnalysisDate", "SampleID"], na_position="last").reset_index(drop=True)


def apply_known_source_corrections(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    oreas_351_zinc_mask = (
        working["SchemeCode"].eq("GE_ICP40Q12")
        & working["SRMSLotCode"].eq("OREAS_351")
        & working["SRMSStandardCode"].eq("OREAS_351")
        & working["Analyte"].eq("ZN")
        & working["UnitCode"].eq("MG_KG")
        & working["TargetValue"].eq(46.99)
    )
    working.loc[oreas_351_zinc_mask, "TargetValue"] = 469900
    working["SourceCorrectionApplied"] = np.where(oreas_351_zinc_mask, "OREAS_351_ZN_TARGET_469900", "")
    return working


def _mad(values: pd.Series) -> float:
    median = values.median()
    return float((values - median).abs().median())


def _rolling_slope(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) < 3:
        return np.nan
    x = np.arange(len(clean), dtype=float)
    return float(np.polyfit(x, clean.to_numpy(dtype=float), 1)[0])


def _run_lengths(values: pd.Series, comparator) -> pd.Series:
    run = 0
    runs = []
    previous = np.nan
    for value in values:
        if pd.notna(previous) and pd.notna(value) and comparator(value, previous):
            run += 1
        else:
            run = 0
        runs.append(run)
        previous = value
    return pd.Series(runs, index=values.index)


def add_known_limit_features(df: pd.DataFrame, config: SRMSConfig) -> pd.DataFrame:
    working = df.copy()
    span = working["UpperLimit"] - working["LowerLimit"]
    fallback_lower = working["LowerLimit"] + span * config.warning_margin_pct
    fallback_upper = working["UpperLimit"] - span * config.warning_margin_pct
    working["WarningLower"] = working["WarningLower"].fillna(fallback_lower)
    working["WarningUpper"] = working["WarningUpper"].fillna(fallback_upper)
    working["WarningThresholdSource"] = np.where(
        df["WarningLower"].notna() & df["WarningUpper"].notna(),
        "source",
        f"fallback_{config.warning_margin_pct:.0%}_inside_acceptance_limits",
    )

    working["Deviation"] = working["Value"] - working["TargetValue"]
    working["DeviationPct"] = safe_divide(working["Deviation"], working["TargetValue"].abs()) * 100
    working["DistanceToLower"] = working["Value"] - working["LowerLimit"]
    working["DistanceToUpper"] = working["UpperLimit"] - working["Value"]
    working["LimitSpan"] = span
    working["WithinRangePct"] = safe_divide(working["Value"] - working["LowerLimit"], span) * 100
    working["DistanceToLowerPctSpan"] = safe_divide(working["DistanceToLower"], span) * 100
    working["DistanceToUpperPctSpan"] = safe_divide(working["DistanceToUpper"], span) * 100

    lower_inclusive = working["LowerLimitInclusive"].fillna("Y").eq("Y")
    upper_inclusive = working["UpperLimitInclusive"].fillna("Y").eq("Y")
    warning_lower_inclusive = working["WarningLowerInclusive"].fillna("Y").eq("Y")
    warning_upper_inclusive = working["WarningUpperInclusive"].fillna("Y").eq("Y")
    lower_fail = np.where(lower_inclusive, working["Value"] < working["LowerLimit"], working["Value"] <= working["LowerLimit"])
    upper_fail = np.where(upper_inclusive, working["Value"] > working["UpperLimit"], working["Value"] >= working["UpperLimit"])
    lower_warning = np.where(warning_lower_inclusive, working["Value"] < working["WarningLower"], working["Value"] <= working["WarningLower"])
    upper_warning = np.where(warning_upper_inclusive, working["Value"] > working["WarningUpper"], working["Value"] >= working["WarningUpper"])

    fail = lower_fail | upper_fail
    warning = lower_warning | upper_warning
    working["LimitStatus"] = np.select([fail, warning], ["FAIL", "WARNING"], default="PASS")
    return working


def add_historical_features(df: pd.DataFrame, config: SRMSConfig) -> pd.DataFrame:
    working = df.sort_values(GROUP_COLUMNS + ["AnalysisDate", "SampleID"], na_position="last").copy()
    grouped = working.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    history = grouped.cumcount()
    prior_value = grouped["Value"].shift(1)

    working["HistoryCount"] = history
    working["HistoricalMean"] = grouped["Value"].transform(lambda s: s.expanding().mean().shift(1))
    working["HistoricalMedian"] = grouped["Value"].transform(lambda s: s.expanding().median().shift(1))
    working["HistoricalStd"] = grouped["Value"].transform(lambda s: s.expanding().std().shift(1))
    working["HistoricalMAD"] = grouped["Value"].transform(lambda s: s.expanding().apply(_mad, raw=False).shift(1))
    working["PreviousValue"] = prior_value
    working["DiffFromPrevious"] = working["Value"] - prior_value
    working["PctDiffFromTarget"] = working["DeviationPct"]

    min_periods = max(2, min(config.min_history, config.rolling_window))
    prior_rolling = grouped["Value"].shift(1).groupby([working[col] for col in GROUP_COLUMNS], dropna=False, sort=False)
    working["RollingMean"] = prior_rolling.transform(lambda s: s.rolling(config.rolling_window, min_periods=min_periods).mean())
    working["RollingStd"] = prior_rolling.transform(lambda s: s.rolling(config.rolling_window, min_periods=min_periods).std())
    working["RollingMeanDiff"] = working["Value"] - working["RollingMean"]
    working["RollingSlope"] = prior_rolling.transform(
        lambda s: s.rolling(config.rolling_window, min_periods=min_periods).apply(_rolling_slope, raw=False)
    )

    working["ConsecutiveIncreases"] = grouped["Value"].transform(lambda s: _run_lengths(s, lambda current, previous: current > previous))
    working["ConsecutiveDecreases"] = grouped["Value"].transform(lambda s: _run_lengths(s, lambda current, previous: current < previous))

    mad_scale = working["HistoricalMAD"].replace(0, np.nan)
    working["RobustZScore"] = 0.6745 * (working["Value"] - working["HistoricalMedian"]) / mad_scale
    zero_mad_changed = (
        working["HistoricalMAD"].abs().le(1e-12)
        & working["Value"].notna()
        & working["HistoricalMedian"].notna()
        & ~np.isclose(working["Value"], working["HistoricalMedian"])
    )
    working.loc[zero_mad_changed, "RobustZScore"] = np.sign(
        working.loc[zero_mad_changed, "Value"] - working.loc[zero_mad_changed, "HistoricalMedian"]
    ) * np.inf
    working.loc[working["HistoryCount"] < config.min_history, "RobustZScore"] = np.nan
    working["InsufficientHistory"] = working["HistoryCount"] < config.min_history
    working["StatisticalAnomalyFlag"] = working["RobustZScore"].abs() >= config.robust_z_threshold

    target_abs = working["TargetValue"].abs()
    working["RollingMeanDiffPctTarget"] = safe_divide(working["RollingMeanDiff"], target_abs) * 100
    working["DiffFromPreviousPctTarget"] = safe_divide(working["DiffFromPrevious"], target_abs) * 100
    working["RollingStdPctTarget"] = safe_divide(working["RollingStd"], target_abs) * 100
    working["RollingSlopePct"] = safe_divide(working["RollingSlope"], target_abs) * 100
    working["RollingMeanShiftPct"] = safe_divide(working["RollingMean"] - working["TargetValue"], target_abs) * 100
    return working.sort_index()


def add_isolation_forest(df: pd.DataFrame, config: SRMSConfig) -> pd.DataFrame:
    working = df.copy()
    feature_columns = IFOREST_FEATURE_COLUMNS
    model_frame = working[feature_columns].replace([np.inf, -np.inf], np.nan)
    eligible = working["HistoryCount"] >= config.min_iforest_history
    usable_features = [column for column in feature_columns if model_frame.loc[eligible, column].notna().any()]

    working["IForestScore"] = np.nan
    working["IForestScoreNorm"] = np.nan
    working["IForestAnomaly"] = False
    working["IForestTrainingMode"] = IFOREST_TRAINING_MODE
    working["IForestFeatureList"] = ", ".join(feature_columns)
    working["IForestSkippedReason"] = ""
    working.loc[~eligible, "IForestSkippedReason"] = "insufficient_history"

    if len(usable_features) == 0 or eligible.sum() < max(config.min_iforest_history, 10):
        working.loc[eligible, "IForestSkippedReason"] = "insufficient_model_features"
        return working

    from sklearn.ensemble import IsolationForest

    frame = model_frame[usable_features].copy()
    train_frame = frame.loc[eligible].copy()
    medians = train_frame.median()
    train_frame = train_frame.fillna(medians).fillna(0)

    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        random_state=config.random_state,
        n_jobs=-1,
    )
    labels = model.fit_predict(train_frame)
    scores = -model.score_samples(train_frame)
    working.loc[eligible, "IForestAnomaly"] = labels == -1
    working.loc[eligible, "IForestScore"] = scores
    working.loc[eligible, "IForestScoreNorm"] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return working


def add_drift_flags(df: pd.DataFrame, config: SRMSConfig) -> pd.DataFrame:
    working = df.copy()
    target_direction = np.sign(working["Deviation"])
    previous_direction = np.sign(working["PreviousValue"] - working["TargetValue"])
    away_from_target = target_direction.eq(previous_direction) & working["Deviation"].abs().gt(
        (working["PreviousValue"] - working["TargetValue"]).abs()
    )
    consecutive_trend = (
        working["ConsecutiveIncreases"].ge(config.drift_consecutive_points - 1)
        | working["ConsecutiveDecreases"].ge(config.drift_consecutive_points - 1)
    )
    strong_slope = working["RollingSlopePct"].abs().ge(config.drift_slope_pct_threshold)
    mean_shift = working["RollingMeanShiftPct"].abs().ge(config.drift_mean_shift_pct_threshold)
    working["DriftFlag"] = (
        working["HistoryCount"].ge(config.min_drift_history)
        & consecutive_trend
        & away_from_target.fillna(False)
        & (strong_slope | mean_shift)
        & working["LimitStatus"].ne("FAIL")
    )
    return working


def add_final_risk(df: pd.DataFrame, config: SRMSConfig) -> pd.DataFrame:
    working = df.copy()
    strong_historical = working["RobustZScore"].abs().ge(config.strong_robust_z_threshold) | working["IForestAnomaly"]
    medium_evidence = working["StatisticalAnomalyFlag"] | working["IForestAnomaly"] | working["DriftFlag"]
    high_evidence = working["LimitStatus"].eq("WARNING") & (strong_historical | working["DriftFlag"])

    working["FinalRiskLevel"] = np.select(
        [
            working["LimitStatus"].eq("FAIL"),
            high_evidence,
            medium_evidence,
        ],
        ["Critical", "High", "Medium"],
        default="Low",
    )

    reasons = []
    for row in working.itertuples(index=False):
        row_reasons = []
        if row.LimitStatus == "FAIL":
            row_reasons.append("outside known acceptance limits")
        elif row.LimitStatus == "WARNING":
            row_reasons.append("close to acceptance limit")
        if bool(getattr(row, "StatisticalAnomalyFlag")):
            row_reasons.append("historical robust z-score anomaly")
        if bool(getattr(row, "IForestAnomaly")):
            row_reasons.append("Isolation Forest anomaly")
        if bool(getattr(row, "DriftFlag")):
            row_reasons.append("drift/trend detected")
        if not row_reasons and bool(getattr(row, "InsufficientHistory")):
            row_reasons.append("inside limits; insufficient historical baseline")
        reasons.append("; ".join(row_reasons) if row_reasons else "inside limits and historically normal")
    working["RiskReason"] = reasons
    return working


def evaluate_results(results: pd.DataFrame, output_dir: Path, config: SRMSConfig) -> Path:
    output_path = output_dir / "srms_evaluation.md"
    label_columns = [column for column in results.columns if "label" in column.lower() or "truth" in column.lower()]
    group_sizes = results.groupby(GROUP_COLUMNS, dropna=False).size()
    risk_summary = create_summary(results)
    eligible_rows = int(results["HistoryCount"].ge(config.min_iforest_history).sum())
    limit_failures = int(results["LimitStatus"].eq("FAIL").sum())
    robust_flags = int(results["StatisticalAnomalyFlag"].sum())
    iforest_flags = int(results["IForestAnomaly"].sum())
    drift_flags = int(results["DriftFlag"].sum())
    agreement = pd.crosstab(results["StatisticalAnomalyFlag"], results["IForestAnomaly"], dropna=False)

    lines = [
        "# SRMS Evaluation Notes",
        "",
        "No reliable reviewed anomaly label is available in the inspected SRMS candidate outputs.",
        "This report therefore avoids artificial accuracy and uses unsupervised QC evidence instead.",
        "",
        "## SRMS Candidate Identification Caveat",
        "",
        "The available ResultSet dataset does not contain an explicit SRMS subtype. Records are treated as SRMS/reference-material standard candidates when they are Standards with a valid reference code, numeric result, target value, and lower/upper acceptance limits. `STD_CODE == SPECIFICATION_CODE` is reported as a diagnostic check, not used as a mandatory hard filter.",
        "",
        "## Dataset and Group Size Diagnostics",
        "",
        f"- Total candidate rows: `{len(results)}`",
        f"- Standard-Scheme-Analyte groups: `{len(group_sizes)}`",
        f"- Median group size: `{group_sizes.median():.1f}`",
        f"- Groups with >= 5 observations: `{int(group_sizes.ge(5).sum())}`",
        f"- Groups with >= 20 observations: `{int(group_sizes.ge(20).sum())}`",
        f"- Groups with >= 50 observations: `{int(group_sizes.ge(50).sum())}`",
        f"- Groups with >= 100 observations: `{int(group_sizes.ge(100).sum())}`",
        f"- Rows eligible for Isolation Forest: `{eligible_rows}` ({eligible_rows / len(results):.1%})",
        "",
        "## Detection Counts",
        "",
        f"- Known acceptance-limit failures: `{limit_failures}`",
        f"- Robust Z-score flags at threshold {config.robust_z_threshold}: `{robust_flags}`",
        f"- Isolation Forest flags at contamination {config.contamination}: `{iforest_flags}`",
        f"- Drift flags: `{drift_flags}`",
        f"- Possible label-like columns found: `{', '.join(label_columns) if label_columns else 'none'}`",
        f"- Isolation Forest training mode: `{IFOREST_TRAINING_MODE}`",
        "",
        "## Risk-Level Distribution",
        "",
        risk_summary.to_markdown(index=False),
        "",
        "## Isolation Forest Features",
        "",
        ", ".join(f"`{column}`" for column in IFOREST_FEATURE_COLUMNS),
        "",
        "## Robust Z-score vs Isolation Forest",
        "",
        agreement.to_markdown(),
        "",
        "## Retrospective Isolation Forest Limitation",
        "",
        "Historical statistical features are calculated from prior observations only. The Isolation Forest is fitted in `batch_retrospective` mode across all eligible candidate rows, so future observations can influence the fitted model, imputation medians, and score normalization for earlier records. This is acceptable for the current retrospective MVP, but not a strict point-in-time production detector.",
        "",
        "## Threshold Assumptions",
        "",
        "- Statistically motivated: Robust Z-score threshold `3.5`, fixed random state for reproducibility.",
        "- QC/business assumptions: known acceptance-limit `FAIL` forces `Critical`; warning fallback uses a `10%` inset when source warning limits are missing.",
        "- Placeholders requiring Datamine validation: strong Robust Z `4.5`, rolling window `5`, minimum history `5`, minimum Isolation Forest history `20`, contamination `0.05`, drift slope/shift/consecutive-movement thresholds.",
        "",
        "## Limitation",
        "",
        "These counts show method agreement and QC triage behaviour, not supervised model accuracy.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def save_group_plots(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib.pyplot as plt

    plot_paths = []
    candidate_groups = (
        results.assign(_risk_weight=results["FinalRiskLevel"].map({"Critical": 4, "High": 3, "Medium": 2, "Low": 1}))
        .groupby(GROUP_COLUMNS, dropna=False)["_risk_weight"]
        .max()
        .sort_values(ascending=False)
        .head(3)
    )

    colors = {"Low": "#4C78A8", "Medium": "#F2CF5B", "High": "#F58518", "Critical": "#D62728"}
    for group_values in candidate_groups.index:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        mask = np.logical_and.reduce([results[column].eq(value) for column, value in zip(GROUP_COLUMNS, group_values)])
        group_df = results.loc[mask].sort_values(["AnalysisDate", "SampleID"])
        if group_df.empty:
            continue
        safe_name = "_".join(str(value).replace("/", "-").replace(" ", "_") for value in group_values)
        path = output_dir / f"srms_group_{safe_name}.png"

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(group_df["AnalysisDate"], group_df["Value"], color="#4C78A8", linewidth=1.4, label="Result")
        ax.axhline(group_df["TargetValue"].median(), color="#222222", linestyle="--", linewidth=1, label="Target")
        ax.axhline(group_df["LowerLimit"].median(), color="#D62728", linestyle=":", linewidth=1, label="Lower limit")
        ax.axhline(group_df["UpperLimit"].median(), color="#D62728", linestyle=":", linewidth=1, label="Upper limit")

        for risk_level, color in colors.items():
            risk_rows = group_df[group_df["FinalRiskLevel"].eq(risk_level)]
            if not risk_rows.empty:
                ax.scatter(risk_rows["AnalysisDate"], risk_rows["Value"], s=28, color=color, label=risk_level, zorder=3)

        drift_rows = group_df[group_df["DriftFlag"]]
        if not drift_rows.empty:
            ax.scatter(
                drift_rows["AnalysisDate"],
                drift_rows["Value"],
                s=80,
                facecolors="none",
                edgecolors="#7F3C8D",
                linewidths=1.5,
                label="Drift",
                zorder=4,
            )

        title = " / ".join(str(value) for value in group_values)
        ax.set_title(f"SRMS result trend: {title}")
        ax.set_xlabel("Analysis date")
        ax.set_ylabel("Measured result")
        ax.legend(loc="best", fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)

    return plot_paths


def create_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = results["FinalRiskLevel"].value_counts().rename_axis("FinalRiskLevel").reset_index(name="Count")
    summary["Percent"] = (summary["Count"] / len(results) * 100).round(2)
    summary["FinalRiskLevel"] = pd.Categorical(
        summary["FinalRiskLevel"],
        categories=["Low", "Medium", "High", "Critical"],
        ordered=True,
    )
    return summary.sort_values("FinalRiskLevel").reset_index(drop=True)


def run_pipeline(raw_df: pd.DataFrame, config: SRMSConfig) -> pd.DataFrame:
    srms = extract_srms_candidates(raw_df)
    corrected = apply_known_source_corrections(srms)
    limited = add_known_limit_features(corrected, config)
    historical = add_historical_features(limited, config)
    modelled = add_isolation_forest(historical, config)
    drifted = add_drift_flags(modelled, config)
    return add_final_risk(drifted, config)


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_cclas_object_property_tsv(args.input):
        history_sheet = args.sheet or config.historical_sheet
        standards, status_summary, risk_summary = run_cclas_standard_pipeline(
            args.input,
            config,
            history_path=args.history,
            sheet_name=history_sheet,
            max_rows=args.max_rows,
        )
        results_path = output_dir / "cclas_standard_detection_results.csv"
        status_summary_path = output_dir / "cclas_standard_status_summary.csv"
        risk_summary_path = output_dir / "cclas_standard_risk_summary.csv"

        order_cclas_standard_columns(standards).to_csv(results_path, index=False)
        status_summary.to_csv(status_summary_path, index=False)
        risk_summary.to_csv(risk_summary_path, index=False)
        report_path = create_cclas_standard_validation_report(
            standards=standards,
            status_summary=status_summary,
            risk_summary=risk_summary,
            output_dir=output_dir,
            input_path=args.input,
        )

        print(f"Processed {len(standards):,} CCLAS Standard/SRMS candidate rows from {args.input}")
        print(f"Saved detailed Standard/SRMS results to {results_path}")
        print(f"Saved Standard status summary to {status_summary_path}")
        print(f"Saved Standard risk summary to {risk_summary_path}")
        print(f"Saved CCLAS validation report to {report_path}")
        return

    raw_df = load_data(args.input, sheet_name=args.sheet, max_rows=args.max_rows)
    results = run_pipeline(raw_df, config)
    summary = create_summary(results)

    ordered_columns = [column for column in OUTPUT_COLUMNS if column in results.columns]
    remaining_columns = [column for column in results.columns if column not in ordered_columns]
    results = results[ordered_columns + remaining_columns]

    results_path = Path(args.results_csv).resolve()
    summary_path = Path(args.summary_csv).resolve()
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    evaluation_path = evaluate_results(results, output_dir, config)
    plot_paths = save_group_plots(results, output_dir)

    print(f"Processed {len(results):,} SRMS candidate rows from {args.input}")
    print(f"Saved detailed results to {results_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved evaluation notes to {evaluation_path}")
    for path in plot_paths:
        print(f"Saved group plot to {path}")


if __name__ == "__main__":
    main()
