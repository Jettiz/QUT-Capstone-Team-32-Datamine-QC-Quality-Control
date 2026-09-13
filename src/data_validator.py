"""
Data Validator Module

Responsibilities:
- Validate data integrity and structure before analysis
- Check for required columns in the dataset
- Handle censored/missing values (especially for blank samples)
- Detect and report data quality issues
- Clean or flag suspicious data points

Key Functions:
- validate_structure(df) -> bool
- validate_blank_data(df) -> dict
- validate_lcs_data(df) -> dict
- validate_duplicate_data(df) -> dict
- validate_replicate_data(df) -> dict
- validate_srm_data(df) -> dict
- check_missing_values(df) -> dict
- check_outliers(df) -> dict

Returns validation report with status and any warnings/errors.
"""

from pathlib import Path

import pandas as pd

# Columns required by LCSDetector.detect(), with the dtype category each
# must satisfy (see control_detector.py's historic-drift pipeline).
_LCS_REQUIRED_COLUMNS = {
    "ANALYTICAL_TYPE": "str",
    "STD_LOT_CODE": "str",
    "STD_CODE": "str",
    "SCHEME_CODE": "str",
    "JOB_CODE": "str",
    "ANALYTE_CODE": "str",
    "ANALYSED_DATE": "datetime",
    "NUMERIC_FINAL_VALUE": "float",
    "INTERNAL_TARGET_VALUE": "float",
    "INTERNAL_MAX_WARNING_VALUE": "float",
    "INTERNAL_MIN_WARNING_VALUE": "float",
}

# STD_CODE values that are not genuine reference-material identifiers (same
# list used by LCSDetector's history grouping and by notebooks/SRMS_LOGIC.md).
_LCS_EXCLUDED_STD_CODES = {"", "Sample", "TSV_BLANK"}

# Columns required to build the replicate/duplicate reference models (see
# notebooks/REP__historical_reference_model.ipynb and
# notebooks/DUP__historical_reference_model.ipynb). Both notebooks load an
# already paired, pre-filtered input (one row per replicate/duplicate pair),
# so there is no ANALYTICAL_TYPE-style subset filter here.
_REP_REQUIRED_COLUMNS = {
    "ANALYTE_CODE": "str",
    "RPD": "float",
    "MEAN_CONC": "float",
    "PRECISION_STATUS": "str",
    "STAT_DL_VALUE": "float",
    "LIM_REP_VALUE": "float",
}
_DUP_REQUIRED_COLUMNS = {
    "ANALYTE_CODE": "str",
    "RPD": "float",
    "MEAN_CONC": "float",
    "PRECISION_STATUS": "str",
    "STAT_DL_DUP_VALUE": "float",
    "LIM_REP_DUP_VALUE": "float",
}

# STAT_DL_(DUP_)VALUE and LIM_REP_(DUP_)VALUE feed
# ALLOWABLE_RPD = 100 * (STAT_DL / MEAN_CONC) + LIM_REP, so a negative value
# is an impossible limit, not just a missing one.
_REP_NONNEGATIVE_COLUMNS = ["STAT_DL_VALUE", "LIM_REP_VALUE"]
_DUP_NONNEGATIVE_COLUMNS = ["STAT_DL_DUP_VALUE", "LIM_REP_DUP_VALUE"]

# Columns required by the SRMS candidate pipeline (see
# src/detectors/srms_detector.py's REQUIRED_COLUMNS / require_columns()).
# PARENT_NUMERIC_FINAL_VALUE is deliberately excluded here: srms_detector.py
# coerces and renames it but never reads it again afterwards.
_SRM_REQUIRED_COLUMNS = {
    "ANALYTICAL_TYPE": "str",
    "STD_LOT_CODE": "str",
    "STD_CODE": "str",
    "JOB_CODE": "str",
    "NUMERIC_FINAL_VALUE": "float",
    "ANALYSED_DATE": "datetime",
    "SCHEME_CODE": "str",
    "ANALYTE_CODE": "str",
    "INTERNAL_MIN_VALUE": "float",
    "INTERNAL_MAX_VALUE": "float",
    "INTERNAL_MIN_INCLUSIVE": "str",
    "INTERNAL_MAX_INCLUSIVE": "str",
    "INTERNAL_MAX_WARNING_VALUE": "float",
    "INTERNAL_MIN_WARNING_VALUE": "float",
    "INTERNAL_MIN_WARNING_INCLUSIVE": "str",
    "INTERNAL_MAX_WARNING_INCLUSIVE": "str",
    "INTERNAL_TARGET_VALUE": "float",
    "UNIT_CODE": "str",
    "SPECIFICATION_CODE": "str",
}


def _check_required_columns(df: pd.DataFrame, required_columns: dict) -> tuple:
    """
    Check presence, null counts, and dtype for each column in
    required_columns. Mirrors the per-column checks in validate_lcs_data.

    Returns (missing_columns, null_counts, dtype_issues).
    """
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        print(f"  Missing required columns: {missing_columns}")

    null_counts = {}
    dtype_issues = {}

    for col, expected_dtype in required_columns.items():
        if col not in df.columns:
            continue

        n_null = int(df[col].isna().sum())
        if n_null > 0:
            null_counts[col] = n_null

        series = df[col].dropna()
        if expected_dtype == "float":
            dtype_ok = pd.api.types.is_numeric_dtype(series)
        elif expected_dtype == "datetime":
            dtype_ok = pd.api.types.is_datetime64_any_dtype(series)
        else:  # "str"
            dtype_ok = not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series)

        actual_dtype = str(df[col].dtype)
        mark = "[ok]" if dtype_ok else "[!!] MISMATCH"
        null_note = f", nulls={n_null}" if n_null else ""
        print(f"    {mark}  {col:<28} expected={expected_dtype:<9} actual={actual_dtype}{null_note}")

        if not dtype_ok:
            dtype_issues[col] = actual_dtype

    return missing_columns, null_counts, dtype_issues


def validate_structure(df: pd.DataFrame, config_dir: str = "config/") -> bool:
    """
    Check if the DataFrame has the expected structure and required columns.

    Reads config/column_config.csv and verifies that every listed
    source_column is present in df. Missing columns marked required=TRUE
    are printed as errors; missing optional columns are printed as warnings.

    Returns True if all required source columns are present, False otherwise.
    """
    config_path = Path(config_dir) / "column_config.csv"
    column_config = pd.read_csv(config_path)

    cols = set(df.columns)
    missing_required = []
    missing_optional = []

    print(f"[DataValidator] -- Structure validation " + "-" * 40)
    for _, row in column_config.iterrows():
        source_column = row["source_column"]
        required = str(row["required"]).strip().upper() == "TRUE"

        if source_column in cols:
            print(f"    [ok]  {source_column}")
        elif required:
            print(f"    [!!] MISSING (required)  {source_column}")
            missing_required.append(source_column)
        else:
            print(f"    [!]   MISSING (optional)  {source_column}")
            missing_optional.append(source_column)

    if missing_required:
        print(f"\n[DataValidator] Missing required columns: {missing_required}")
    if missing_optional:
        print(f"[DataValidator] Missing optional columns: {missing_optional}")

    return len(missing_required) == 0


def validate_blank_data(df: pd.DataFrame) -> dict:
    """
    Validate blank sample data:
    - Check for censored values (< LOD)
    - Verify numeric columns
    - Detect obvious data entry errors
    """
    pass


def validate_lcs_data(df: pd.DataFrame) -> dict:
    """
    Validate LCS (Control) data ahead of LCSDetector.

    1. Filters df to the Control/LCS subset: ANALYTICAL_TYPE == "Standard"
       with a genuine STD_CODE (excludes blank/"Sample"/"TSV_BLANK" --
       STD_LOT_CODE == "Sample" is a labelling quirk of STD_CODE ==
       "OREAS_502C", not a distinct category; see
       notebooks/LCS_drift_detection_historic.ipynb's design doc).
    2. For each column LCSDetector requires, checks the subset has no
       missing (null) values and the column's dtype matches what the
       detector expects (str / float / datetime).
    """
    print(f"[DataValidator] -- LCS (Control) data validation " + "-" * 40)

    missing_columns = [c for c in _LCS_REQUIRED_COLUMNS if c not in df.columns]
    if "ANALYTICAL_TYPE" in df.columns and "STD_CODE" in df.columns:
        std_code = df["STD_CODE"].fillna("").astype(str).str.strip()
        control_df = df[
            (df["ANALYTICAL_TYPE"] == "Standard") & (~std_code.isin(_LCS_EXCLUDED_STD_CODES))
        ]
    else:
        control_df = df.iloc[0:0]

    print(f"  Control/LCS rows (ANALYTICAL_TYPE=='Standard' & STD_CODE not excluded): {len(control_df):,}")

    if missing_columns:
        print(f"  Missing required columns: {missing_columns}")

    null_counts = {}
    dtype_issues = {}

    for col, expected_dtype in _LCS_REQUIRED_COLUMNS.items():
        if col not in control_df.columns:
            continue

        n_null = int(control_df[col].isna().sum())
        if n_null > 0:
            null_counts[col] = n_null

        series = control_df[col].dropna()
        if expected_dtype == "float":
            dtype_ok = pd.api.types.is_numeric_dtype(series)
        elif expected_dtype == "datetime":
            dtype_ok = pd.api.types.is_datetime64_any_dtype(series)
        else:  # "str"
            dtype_ok = not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series)

        actual_dtype = str(control_df[col].dtype)
        mark = "[ok]" if dtype_ok else "[!!] MISMATCH"
        null_note = f", nulls={n_null}" if n_null else ""
        print(f"    {mark}  {col:<28} expected={expected_dtype:<9} actual={actual_dtype}{null_note}")

        if not dtype_ok:
            dtype_issues[col] = actual_dtype

    status = not missing_columns and not null_counts and not dtype_issues

    if status:
        print("  Result: OK")
    else:
        print(f"  Result: FAILED (missing_columns={missing_columns}, null_counts={null_counts}, dtype_issues={dtype_issues})")

    return {
        "status": status,
        "n_rows": len(control_df),
        "missing_columns": missing_columns,
        "null_counts": null_counts,
        "dtype_issues": dtype_issues,
    }


def _validate_duplicate_like(
    df: pd.DataFrame, required_columns: dict, nonnegative_columns: list, label: str
) -> dict:
    """
    Shared implementation for validate_duplicate_data / validate_replicate_data:
    checks required_columns for missing/null/dtype issues, then flags negative
    values in nonnegative_columns (the STAT_DL/LIM_REP limit columns, which
    feed ALLOWABLE_RPD = 100 * (STAT_DL / MEAN_CONC) + LIM_REP and so cannot
    be negative).
    """
    print(f"[DataValidator] -- {label} data validation " + "-" * 40)

    missing_columns, null_counts, dtype_issues = _check_required_columns(df, required_columns)

    invalid_limits = {}
    for col in nonnegative_columns:
        if col not in df.columns:
            continue
        n_negative = int((pd.to_numeric(df[col], errors="coerce").dropna() < 0).sum())
        if n_negative:
            invalid_limits[col] = n_negative
    if invalid_limits:
        print(f"  Invalid limit values (negative): {invalid_limits}")

    status = not missing_columns and not null_counts and not dtype_issues and not invalid_limits

    if status:
        print("  Result: OK")
    else:
        print(
            f"  Result: FAILED (missing_columns={missing_columns}, null_counts={null_counts}, "
            f"dtype_issues={dtype_issues}, invalid_limits={invalid_limits})"
        )

    return {
        "status": status,
        "n_rows": len(df),
        "missing_columns": missing_columns,
        "null_counts": null_counts,
        "dtype_issues": dtype_issues,
        "invalid_limits": invalid_limits,
    }


def validate_duplicate_data(df: pd.DataFrame) -> dict:
    """
    Validate duplicate-pair data ahead of the duplicate reference-model
    analysis (notebooks/DUP__historical_reference_model.ipynb).

    Checks ANALYTE_CODE, RPD, MEAN_CONC, and PRECISION_STATUS (used to
    filter/group the reference population) plus the duplicate-specific
    limit columns STAT_DL_DUP_VALUE/LIM_REP_DUP_VALUE (used to compute the
    allowable-RPD curve), for missing columns, nulls, dtype mismatches, and
    negative limit values.
    """
    return _validate_duplicate_like(df, _DUP_REQUIRED_COLUMNS, _DUP_NONNEGATIVE_COLUMNS, "Duplicate")


def validate_replicate_data(df: pd.DataFrame) -> dict:
    """
    Validate replicate-pair data ahead of the replicate reference-model
    analysis (notebooks/REP__historical_reference_model.ipynb).

    Same checks as validate_duplicate_data, against the replicate-specific
    limit columns STAT_DL_VALUE/LIM_REP_VALUE.
    """
    return _validate_duplicate_like(df, _REP_REQUIRED_COLUMNS, _REP_NONNEGATIVE_COLUMNS, "Replicate")


def validate_srm_data(df: pd.DataFrame) -> dict:
    """
    Validate SRM (reference-material) data ahead of the SRMS candidate
    pipeline (src/detectors/srms_detector.py).

    1. Filters df to the same Standard/reference-material subset
       validate_lcs_data uses: ANALYTICAL_TYPE == "Standard" with a genuine
       STD_CODE -- this matches srms_detector.py's extract_srms_candidates
       filter.
    2. Checks each column required by srms_detector.py's require_columns()
       for missing values and expected dtype (PARENT_NUMERIC_FINAL_VALUE is
       excluded -- it's loaded but never used afterwards).
    3. Flags an impossible acceptance-limit span (INTERNAL_MIN_VALUE >=
       INTERNAL_MAX_VALUE), since UpperLimit - LowerLimit is used as a
       divisor throughout the detector's distance/warning-fallback
       calculations.
    """
    print(f"[DataValidator] -- SRM (Reference Material) data validation " + "-" * 40)

    if "ANALYTICAL_TYPE" in df.columns and "STD_CODE" in df.columns:
        std_code = df["STD_CODE"].fillna("").astype(str).str.strip()
        srm_df = df[
            (df["ANALYTICAL_TYPE"] == "Standard") & (~std_code.isin(_LCS_EXCLUDED_STD_CODES))
        ]
    else:
        srm_df = df.iloc[0:0]

    print(f"  SRM rows (ANALYTICAL_TYPE=='Standard' & STD_CODE not excluded): {len(srm_df):,}")

    missing_columns, null_counts, dtype_issues = _check_required_columns(srm_df, _SRM_REQUIRED_COLUMNS)

    invalid_limits = {}
    if "INTERNAL_MIN_VALUE" in srm_df.columns and "INTERNAL_MAX_VALUE" in srm_df.columns:
        lower = pd.to_numeric(srm_df["INTERNAL_MIN_VALUE"], errors="coerce")
        upper = pd.to_numeric(srm_df["INTERNAL_MAX_VALUE"], errors="coerce")
        n_bad_span = int((upper <= lower).sum())
        if n_bad_span:
            invalid_limits["INTERNAL_MIN_VALUE/INTERNAL_MAX_VALUE"] = n_bad_span
    if invalid_limits:
        print(f"  Invalid limit values (lower >= upper): {invalid_limits}")

    status = not missing_columns and not null_counts and not dtype_issues and not invalid_limits

    if status:
        print("  Result: OK")
    else:
        print(
            f"  Result: FAILED (missing_columns={missing_columns}, null_counts={null_counts}, "
            f"dtype_issues={dtype_issues}, invalid_limits={invalid_limits})"
        )

    return {
        "status": status,
        "n_rows": len(srm_df),
        "missing_columns": missing_columns,
        "null_counts": null_counts,
        "dtype_issues": dtype_issues,
        "invalid_limits": invalid_limits,
    }

def validate_matrix_spike_dupl_data(df: pd.DataFrame) -> dict:
    """
    Validate matrix spike duplicate data:
    - Check for duplicate spike entries
    """
    pass

def validate_matrix_spike_data(df: pd.DataFrame) -> dict:
    """
    Validate matrix spike data:
    - Check spike concentrations are defined
    """
    pass

