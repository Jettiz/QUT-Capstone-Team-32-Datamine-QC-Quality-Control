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
- validate_srm_data(df) -> dict
- check_missing_values(df) -> dict
- check_outliers(df) -> dict

Returns validation report with status and any warnings/errors.
"""

from pathlib import Path

import pandas as pd

# Columns required by LCSDetector.detect()/detect_drift(), with the dtype
# category each must satisfy (see lcs_detector.py required_cols).
_LCS_REQUIRED_COLUMNS = {
    "ANALYTICAL_TYPE": "str",
    "STD_LOT_CODE": "str",
    "JOB_CODE": "str",
    "ANALYTE_CODE": "str",
    "ANALYSED_DATE": "datetime",
    "NUMERIC_FINAL_VALUE": "float",
    "INTERNAL_TARGET_VALUE": "float",
    "INTERNAL_MAX_WARNING_VALUE": "float",
    "INTERNAL_MIN_WARNING_VALUE": "float",
}


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
       and STD_LOT_CODE == "Sample" (same rule as check_classifier's "lcs" check).
    2. For each column LCSDetector requires, checks the subset has no
       missing (null) values and the column's dtype matches what the
       detector expects (str / float / datetime).
    """
    print(f"[DataValidator] -- LCS (Control) data validation " + "-" * 40)

    missing_columns = [c for c in _LCS_REQUIRED_COLUMNS if c not in df.columns]
    if "ANALYTICAL_TYPE" in df.columns and "STD_LOT_CODE" in df.columns:
        control_df = df[
            (df["ANALYTICAL_TYPE"] == "Standard") & (df["STD_LOT_CODE"] == "Sample")
        ]
    else:
        control_df = df.iloc[0:0]

    print(f"  Control/LCS rows (ANALYTICAL_TYPE=='Standard' & STD_LOT_CODE=='Sample'): {len(control_df):,}")

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


def validate_duplicate_data(df: pd.DataFrame) -> dict:
    """
    Validate duplicate pairs:
    - Ensure samples are properly paired

    """
    pass


def validate_srm_data(df: pd.DataFrame) -> dict:
    """
    Validate SRM data:
    - Check certified values are defined
    """
    pass

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

