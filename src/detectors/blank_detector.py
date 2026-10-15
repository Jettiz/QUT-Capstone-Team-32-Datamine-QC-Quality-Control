'''

Comment

'''

# Neccesary Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.__config__ import CONFIG
import pandas as pd
import yaml
import statistics


from scipy.stats import theilslopes

from .base_detector import LCSAnomalyResult
from ..data_validator import select_lcs_rows
from ..qc_status import FAIL, PASS, STATE_RANK, WARNING
from ..utils import atomic_write_csv, load_csv_or_empty


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# -- Grouping --------------------

PROJECT_ROOT = Path(__file__ ).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "b_outputs"


class B_CONFIG:

    GROUP_COLS: tuple = ["STD_LOT_CODE", "SCHEME_CODE", "ANALYTE_CODE"],

    IGNORED_STATUSES: tuple = {"IgnoredUpperFailure", "IgnoredLowerFailure"},

REQUIRED_COLUMNS = [
    "STD_CODE", "ANALYTE_CODE", "SCHEME_CODE", "JOB_CODE", "NUMERIC_FINAL_VALUE" "ANALYSED_DATE", "STANDARD_STATUS", "INTERNAL_MIN_VALUE", "INTERNAL,_MAX_VALUE", "INTERNAL_MIN_INCLUSIVE", "INTERNAL_MAX_INCLUSIVE", "INTERNAL_TARGET_VALUE", "UNIT_CODE", "SPECIFICATION_CODE", "INSTRUMENT_ID",
]

OUTPUT_COLUMNS = [
    "STD_CODE", "SCHEME_CODE", "ANALYTE_CODE", "JOB_CODE", "ANALYSED_DATE", "IS_IGNORED", "STANDARD_STATUS", "NUMERIC_FINAL_VALUE", "INTERNAL_MIN_VALUE", "INTERNAL,_MAX_VALUE", "INTERNAL_MIN_INCLUSIVE", "INTERNAL_MAX_INCLUSIVE", "INTERNAL_TARGET_VALUE", "UNIT_CODE", "SPECIFICATION_CODE", "INSTRUMENT_ID", "OUTSIDE_THRESHHOLD", "P_ZTEST", "P_ANOM", "ROLL_MEDIAN_WEEK", "ROLLING_CHANGE_WEEK", "ROLL_STD_WEEK", "ROLL_Z_SCORE_WEEK", "IS_DRIFT", "ANOMALY", "FINAL_RISK", "REASON",
]


# -- DEFS --------------------


def load_b_data(path_str: str, sheet_name: str = "Blank (B) Assessment") -> pd.DataFrame:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, low_memory=False, keep_default_na=False)
    else:
        raw = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", keep_default_na=False)
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    unnamed = [c for c in raw.columns if c.strip() == "" or c.startswith("UNNAMED")]
    if unnamed:
        raw = raw.drop(columns=unnamed)
    return raw

def filter_b_data(raw: pd.DataFrame) -> pd.DataFrame:

    required = [
        "ANALYTIC_CODE",
    ]

    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = raw.copy()

    for col in required:
        cleaned[col] = (
            cleaned[col].astype(str).str.strip()
        )

    mask = (
        cleaned["ANALYTIC_CODE"].notna() & cleaned["ANALYTIC_CODE"].ne("") & ~cleaned["ANALYTIC_CODE"].str.casefold().eq("na")
    )

    return cleaned.loc[mask.fillna(False)].copy()

def validate_b_data(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def prepare_b_data(df: pd.DataFrame, cfg: B_CONFIG) -> pd.DataFrame:
    numeric_cols = [
        "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE", "INTERNAL_MIN_VALUE", "INTERNAL_MAX_VALUE"
    ]

    df = df.copy()

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop(['PRECISION_STATUS', 'INTERNAL_MAX_WARNING_VALUE', 'INTERNAL_MIN_WARNING_VALUE', 'INTERNAL_MAX_WARNING_INCLUSIVE', 'INTERNAL_MIN_WARNING_INCLUSIVE', 'PARENT_NUMERIC_FINAL_VALUE'], axis=1, inplace=False)
    df = df.dropna(subset=["ANALYTE_CODE"])
    df["ANALYSED_DATE"] = pd.to_datetime(df["ANALYSED_DATE"], dayfirst=True)
    df["IS_IGNORED"] = df["STANDRD_STATUS"].isin(cfg.IGNORED_STATUSES)
    df =  df.sort_values(list(cfg.GROUP_COLS) + ["ANALYSED_DATE"]).reset_index(drop=True)
    return df

def b_features(df: pd.DataFrame, cfg: CONFIG) -> pd.DataFrame:
    gc = list(cfg.GROUP_COLS)

    # Target Deviation
    df["TARGET_DEV"] = df["NUMERIC_FINAL_VALUE"] - df["INTERNAL_TARGET_VALUE"]

    # Group Median
    df["GROUP_MEDIAN"] = (df.groupbyt(gc)["NUMERIC_FINAL_VALUE"].transform("median"))

    # ABS Median Deviation
    df["ABS_MEDIAN_DEVIATION"] = (
        df["NUMERIC_FINAL_VALUE"]
        - df["GROUP_MEDIAN"]
    ).abs()

    # Group MAD
    df["GROUP_MAD"] = (
        df.groupby(gc)["ABS_MEDIAN_DEVIATION"].transform("median")
    )

    # Rolling Median
    df["ROLL_MEDIAN"] = (
        df.groupby(gc, dropna=True)["NUMERIC_FINAL_VALUE"].transform(lambda x: x.rolling(window=10, min_periods=3).median())
    )

    # Rolling Change
    df["ROLLING_CHANGE"] = (df["ROLL_MEDIAN"] - df["GROUP_MEDIAN"])

    # Z Score
    df["ROB_Z_SCORE"] = np.where(
        df["GROUP_MAD"] > 0,
        0.6745 * (df["NUMERIC_FINAL_VALUE"] - df["GROUP_MEDIAN"])
        / df["GROUP_MEDIAN"],
        np.nan
    )

    return df

def b_point_detection(df: pd.DataFrame, cfg: CONFIG) -> pd.DataFrame:
    df["OUTSIDE_TRESHHOLD"] = (
        df["NUMERIC_FINAL_VALUE"] > df["INTERNAL_MAX_VALUE"]) | (df["NUMERIC_FINAL_VALUE"] < df["INTERNAL_MIN_VALUE"])

    threshhold_df = df["OUTSIDE_TRESHHOLD"] == True

    # Z Score Test
    df["P_ZTest"] = (
        df["ROB_Z_SCORE"] > 3
    )

    ZSCORE_df = df["P_ZTEST"] == True

    df["P_ANOM"] = df["OUTSIDE_TRESHHOLD"] | df["P_ZTEST"]

    P_df = df["P_ANOM"] == True

    return df

def b_drift_detection(df: pd.DataFrame, cfg: CONFIG) -> pd.DataFrame:

    gc = list(cfg.GROUP_COLS)
    
    df = df.sort_values("ANALYSED_DATE")
    df = df.reset_index()
    df = df.set_index("ANALYSED_DATE")

    df["ROLL_MEDIAN_WEEK"] = (
        df.groupby(gc, dropna=True)["NUMERIC_FINAL_VALUE"]
        .rolling(window="7D", min_periods=1)
        .std()
        .reset_index(level=gc, drop=True)
    )

    df["ROLLING_CHANGE_WEEK"] = (
        df["ROLL_MEDIAN_WEEK"] - df["GROUP_MEDIAN"]
    )

    rolling_std = (
        df.groupby(gc, dropna=True)["NUMERIC_FINAL_VALUE"]
        .rolling(window="7D", min_periods=1)
        .std()
        .reset_index(level=gc, drop=True)
    )

    df["ROLL_STD_WEEK"] = rolling_std.values

    df["ROLL_Z_SCORE_WEEK"] = (
        (df["NUMERIC_FINAL_VALUE"] - df["ROLL_MEDIAN_WEEK"])
        / df["ROLL_STD_WEEK"]
    ).abs()

    df["IS_DFRIFT"] = df["ROLL_Z_SCORE_WEEK"] > 10.0

    return df

def b_anom_detector(df: pd.DataFrame, cfg: CONFIG) -> pd.DataFrame:
    df["ANOMALY"] = (df["IS_DRIFT"] == True) | (df["P_ANOM"] == True)

    return df

# ----

def _assign_risk(ANOMALY: bool, IS_DRIFT: bool, IS_IGNORED: bool, P_ANOM: bool) -> str:
    if IS_IGNORED:
        return "Ignored"
    if ANOMALY == True and IS_DRIFT == True:
        return "Drift Anomaly"
    if ANOMALY == True and P_ANOM == True:
        return "Point Anomaly"
    if ANOMALY == True and IS_DRIFT == True and P_ANOM == True:
        return "Point and Drift Anomaly"
    if ANOMALY == False:
        return "No Anomaly"

def _assign_reason(
        IS_IGNORED: bool,
        FINAL_RISK: str
) -> str:
    if IS_IGNORED:
        return "Ignored"
    if FINAL_RISK == "Drift Anomaly":
        return "Drift Anomaly Detected"
    if FINAL_RISK == "Point Anomaly":
        return "Point Anomaly Detected"
    if FINAL_RISK == "Point and Drift Anomaly":
        return "Point and Drift Anomaly Detected"
    if FINAL_RISK == "No Anomaly":
        return "None"

# ----

def b_apply_risk(df: pd.DataFrame, cfg: CONFIG) -> pd.DataFrame:
    df["FINAL_RISK"] = [
        _assign_risk(row["ANOMALY"], row["IS_DRIFT"], row["IS_IGNORED"], row["P_ANOM"])
        for _, row in df.iterrows()
    ]

    df["REASON"] = [
        _assign_reason(row["IS_IGNORED"], row["FINAL_RISK"])
    ]

    return df


def run_b_detection(
        df: pd.DataFrame,
        cfg: CONFIG | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    validate_b_data(df)
    df = prepare_b_data(df.copy(), cfg)
    df = b_features(df, cfg)
    df = b_point_detection(df, cfg)
    df = b_drift_detection(df, cfg)
    df = b_anom_detector(df, cfg)

    df = b_apply_risk(df, cfg)

    out_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    results = df[out_cols].copy()

    return results

def export_results(
    results: pd.DataFrame,
    output_dir,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    full_path = output_dir / "b_detection_results.csv"
    anom_order = pd.CategoricalDtype(
        categories=["Point and Drift Anomaly", "Point Anomaly", "Drift Anomaly", "No Anomaly"]
    )

    full_export = results.copy()
    full_export["_ANOMALY"] = full_export["FINAL_RISK"].astype(anom_order)
    full_export = full_export.sort_values(
        ["_ANOMALY"], ascending=[True, False], na_position="last"
    ).drop(columns="_ANOMALY")
    full_export.to_csv(full_path, index=False)
    paths["full"] = full_path

    flagged = results[results["FINAL_RISK"].isin(["Point and Drift Anomaly", "Point Anomaly", "Drift Anomaly"])].copy()
    flagged_path = output_dir / "b_detection_flagged.csv"
    flagged["_ANOMALY"] = flagged["FINAL_RISK"].astype(anom_order)
    flagged = flagged.sort_values(
        ["_ANOMALY"], ascending=[True, False], na_position="last"
    ).drop(columns="_ANOMALY")
    flagged.to_csv(flagged_path, index=False)
    paths["flagged"] = flagged_path

    return paths

def print_summary(results: pd.DataFrame) -> None:
    anom_counts = results["FINAL_RISK"].value_counts()
    print("\nBlank Detection Summary")
    print("=" * 40)
    print(f"Total records:     {len(results):,}")
    for level in ["Point and Drift Anomaly", "Point Anomaly", "Drift Anomaly", "No Anomaly", "Ignored"]:
        print(f"  {level:<12}  {anom_counts.get(level, 0):>5}")
    print(f"\nIF anomalies:      {results["ANOMALY"].sum():,}")


# -- MAIN --

def main():

    input_path = "data/raw/QC_Anomaly_Training_Data_v3.xlsx"
    output_path = "b_outputs"
    sheet_name = "Blank (B) Assessment"

    print(f"Loading MS data from: {input_path}")

    raw = load_b_data(
        input_path,
        sheet_name=sheet_name,
    )

    b = filter_b_data(raw)

    print(f"Blank records after filter: {len(b):,}")
    print(f"Unique analytes: {b['ANALYTE_CODE'].nunique():,}")

    results = run_b_detection(b, cfg)

    paths = export_results(
        results, output_dir,
    )

    print_summary(results)

    print("\nOutputs written:")
    for name, path in paths.items():
        print(f"  {name:<15} {path}")

if __name__ == "__main__":
    main()
