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
- validate_matrix_spike_data(df) -> dict
- check_missing_values(df) -> dict
- check_outliers(df) -> dict

Returns validation report with status and any warnings/errors.

Column-naming convention
-------------------------
Every validator checks the LOWERCASE internal column names produced by
data_loader.load_qc_data() (e.g. analytical_type, std_lot_code,
analyte_code, measured_value, rpd, mean_conc -- see config/column_config.csv
and the loader's _derive_precision_metrics() for rpd/mean_conc). Field names
are standardised inside the product: raw export names only exist up to the
loader.

Note: src/detectors/srms_detector.py (owned by the SRMS teammate) still
consumes the raw uppercase CCLAS schema internally; validate_srm_data()
already validates the internal-schema subset that will be assigned to it.

Sample routing (which rows go to which QC method)
--------------------------------------------------
This module is the single place where Standard rows are split between the
two methods that analyse reference materials -- see select_lcs_rows() and
select_srm_rows(). Detectors and the POC call these functions instead of
re-implementing the split.
"""

from pathlib import Path

import pandas as pd

# Columns required by LCSDetector.detect() (internal names), with the dtype
# category each must satisfy (see control_detector.py's REQUIRED_COLUMNS).
_LCS_REQUIRED_COLUMNS = {
    "analytical_type": "str",
    "std_lot_code": "str",
    "std_code": "str",
    "scheme_code": "str",
    "job_code": "str",
    "analyte_code": "str",
    "analysed_date": "datetime",
    "measured_value": "float",
    "target_value": "float",
    "limit_max": "float",
    "limit_min": "float",
    "limit_max_warning": "float",
    "limit_min_warning": "float",
}

# ── Sample routing ───────────────────────────────────────────────────────────
# Both LCS (Control) and SRMS analyse ANALYTICAL_TYPE == "Standard" rows.
# A Standard row whose STD_LOT_CODE is "Sample" is a laboratory control
# sample (LCS); every other Standard row is a reference-material standard for
# SRMS. Comparisons are case-insensitive and ignore surrounding whitespace.
_STANDARD_ANALYTICAL_TYPE = "standard"
_LCS_STD_LOT_CODE = "sample"

# Columns required to build the replicate/duplicate reference models (see
# notebooks/REP__historical_reference_model.ipynb and
# notebooks/DUP__historical_reference_model.ipynb, and
# notebooks/REP__distribution_analysis.ipynb / DUP__distribution_analysis.ipynb
# for where rpd/mean_conc actually come from). Lowercase internal names --
# data_loader.load_qc_data() renames the raw CCLAS columns AND derives
# rpd/mean_conc (see its _derive_precision_metrics()); this validator checks
# its output, not the raw file. Both notebooks load an already paired,
# pre-filtered input (one row per replicate/duplicate pair), so there is no
# analytical_type-style subset filter here either.
_REP_REQUIRED_COLUMNS = {
    "analyte_code": "str",
    "rpd": "float",
    "mean_conc": "float",
    "precision_status": "str",
    "stat_detection_limit": "float",
    "limiting_repeatability": "float",
}
_DUP_REQUIRED_COLUMNS = {
    "analyte_code": "str",
    "rpd": "float",
    "mean_conc": "float",
    "precision_status": "str",
    "stat_detection_limit_dup": "float",
    "limiting_repeatability_dup": "float",
}

# stat_detection_limit(_dup)/limiting_repeatability(_dup) feed
# ALLOWABLE_RPD = 100 * (STAT_DL / MEAN_CONC) + LIM_REP, so a negative value
# is an impossible limit, not just a missing one.
_REP_NONNEGATIVE_COLUMNS = ["stat_detection_limit", "limiting_repeatability"]
_DUP_NONNEGATIVE_COLUMNS = ["stat_detection_limit_dup", "limiting_repeatability_dup"]

# Columns required for Blank sample validation (see notebooks/BLANK_updated.ipynb
# plus repo-owner clarification on data/raw/ResultSet.csv's real Blank rows):
# NUMERIC_FINAL_VALUE is the found value, INTERNAL_MIN_VALUE/INTERNAL_MAX_VALUE
# are the failure limits, INTERNAL_TARGET_VALUE is the expected value, and the
# other grouping columns match LCS/Control's. No censored-value/LOD check is
# included -- there is no real detection-limit column for Blank anywhere in
# this data (BLANK_updated.ipynb's own "<LOD" check is a text-pattern scan
# that found zero matches and isn't a real business rule). Warning-limit
# columns and instrument are deliberately excluded from required_columns:
# confirmed 0% and ~0% (1/17,997) populated respectively for Blank rows in
# ResultSet.csv.
_BLANK_REQUIRED_COLUMNS = {
    "analytical_type": "str",
    "std_lot_code": "str",
    "std_code": "str",
    "scheme_code": "str",
    "job_code": "str",
    "analyte_code": "str",
    "analysed_date": "datetime",
    "measured_value": "float",
    "target_value": "float",
    "limit_min": "float",
    "limit_max": "float",
    "unit_code": "str",
}

# Columns required for Matrix Spike validation (see notebooks/MS_Detection.ipynb
# and data/raw/QC_Anomaly_Training_Data_v2.xlsx's "SPK(MS) Assessment" sheet --
# confirmed to share the identical 27-column ResultSet.csv/SRM schema, plus
# QC_TYPE). Mirrors _SRM_REQUIRED_COLUMNS closely since the real data is
# structurally the same. parent_value (PARENT_NUMERIC_FINAL_VALUE) and
# instrument_id are deliberately excluded: confirmed 0% populated in the real
# MS export, matching the notebook's own documented finding that recovery %
# "cannot be computed" from this data -- detection there uses the same
# target/limit transform as LCS/SRM instead.
_MS_REQUIRED_COLUMNS = {
    "analytical_type": "str",
    "std_lot_code": "str",
    "std_code": "str",
    "job_code": "str",
    "scheme_code": "str",
    "analyte_code": "str",
    "analysed_date": "datetime",
    "measured_value": "float",
    "target_value": "float",
    "limit_min": "float",
    "limit_max": "float",
    "limit_min_inclusive": "str",
    "limit_max_inclusive": "str",
    "limit_max_warning": "float",
    "limit_min_warning": "float",
    "limit_min_warning_inclusive": "str",
    "limit_max_warning_inclusive": "str",
    "unit_code": "str",
    "specification_code": "str",
}

# Columns required by the SRMS candidate pipeline (see
# src/detectors/srms_detector.py's REQUIRED_COLUMNS / require_columns()),
# expressed as the internal names they map to in config/column_config.csv.
# parent_value (PARENT_NUMERIC_FINAL_VALUE) is deliberately excluded here:
# srms_detector.py coerces and renames it but never reads it again afterwards.
_SRM_REQUIRED_COLUMNS = {
    "analytical_type": "str",
    "std_lot_code": "str",
    "std_code": "str",
    "job_code": "str",
    "measured_value": "float",
    "analysed_date": "datetime",
    "scheme_code": "str",
    "analyte_code": "str",
    "limit_min": "float",
    "limit_max": "float",
    "limit_min_inclusive": "str",
    "limit_max_inclusive": "str",
    "limit_max_warning": "float",
    "limit_min_warning": "float",
    "limit_min_warning_inclusive": "str",
    "limit_max_warning_inclusive": "str",
    "target_value": "float",
    "unit_code": "str",
    "specification_code": "str",
}


def _normalised(series: pd.Series) -> pd.Series:
    """Lower-cased, whitespace-trimmed string view of a column (missing -> "")."""
    return series.fillna("").astype(str).str.strip().str.casefold()


def select_standard_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows with analytical_type == "Standard" (case-insensitive). Returns an
    empty slice if the column is absent.
    """
    if "analytical_type" not in df.columns:
        return df.iloc[0:0]
    return df[_normalised(df["analytical_type"]).eq(_STANDARD_ANALYTICAL_TYPE)]


def _lcs_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of the rows routed to LCS (see select_lcs_rows)."""
    if "analytical_type" not in df.columns or "std_lot_code" not in df.columns:
        return pd.Series(False, index=df.index)
    is_standard = _normalised(df["analytical_type"]).eq(_STANDARD_ANALYTICAL_TYPE)
    return is_standard & _normalised(df["std_lot_code"]).eq(_LCS_STD_LOT_CODE)


def select_lcs_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows routed to the LCS (Control) analysis: Standard rows whose
    std_lot_code is "Sample" (case-insensitive, trimmed).

    This is the single source of truth for the LCS side of the Standard
    split -- LCSDetector.detect(), validate_lcs_data() and the POC all call it.
    """
    return df[_lcs_mask(df)]


def select_srm_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows routed to the SRMS analysis: every Standard row that is NOT routed
    to LCS (i.e. std_lot_code is anything other than "Sample", including
    missing). Non-Standard rows never reach either method.
    """
    if "analytical_type" not in df.columns:
        return df.iloc[0:0]
    is_standard = _normalised(df["analytical_type"]).eq(_STANDARD_ANALYTICAL_TYPE)
    return df[is_standard & ~_lcs_mask(df)]


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
    Validate Blank sample data ahead of a future blank-anomaly detector
    (src/detectors/blank_detector.py, currently an empty stub).

    1. Filters df to the Blank subset: analytical_type == "blank"
       (case-insensitive, matching notebooks/BLANK_updated.ipynb's own
       `.str.casefold().eq("blank")` filter).
    2. For each column required, checks the subset has no missing (null)
       values and the column's dtype matches what's expected.

    No censored-value/limit-of-detection check is performed -- see
    _BLANK_REQUIRED_COLUMNS' comment for why.
    """
    print(f"[DataValidator] -- Blank data validation " + "-" * 40)

    if "analytical_type" in df.columns:
        analytical_type = df["analytical_type"].fillna("").astype(str).str.strip().str.casefold()
        blank_df = df[analytical_type.eq("blank")]
    else:
        blank_df = df.iloc[0:0]

    print(f"  Blank rows (analytical_type=='blank'): {len(blank_df):,}")

    missing_columns, null_counts, dtype_issues = _check_required_columns(blank_df, _BLANK_REQUIRED_COLUMNS)

    status = not missing_columns and not null_counts and not dtype_issues

    if status:
        print("  Result: OK")
    else:
        print(f"  Result: FAILED (missing_columns={missing_columns}, null_counts={null_counts}, dtype_issues={dtype_issues})")

    return {
        "status": status,
        "n_rows": len(blank_df),
        "missing_columns": missing_columns,
        "null_counts": null_counts,
        "dtype_issues": dtype_issues,
    }


def validate_lcs_data(df: pd.DataFrame) -> dict:
    """
    Validate LCS (Control) data (data_loader.load_qc_data()'s internal
    schema) ahead of LCSDetector.

    1. Filters df to the rows routed to LCS by select_lcs_rows(): Standard
       rows whose std_lot_code is "Sample".
    2. For each column LCSDetector requires, checks the subset has no
       missing (null) values and the column's dtype matches what the
       detector expects (str / float / datetime).

    Note that LCSDetector itself tolerates (and reports) rows with missing
    limits by skipping them, so null_counts here describe rows that will be
    skipped rather than a reason the analysis cannot run.
    """
    print(f"[DataValidator] -- LCS (Control) data validation " + "-" * 40)

    control_df = select_lcs_rows(df)
    print(f"  Control/LCS rows (analytical_type=='Standard' & std_lot_code=='Sample'): {len(control_df):,}")

    missing_columns, null_counts, dtype_issues = _check_required_columns(control_df, _LCS_REQUIRED_COLUMNS)

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
    Validate duplicate-pair data (data_loader.load_qc_data()'s internal
    schema) ahead of the duplicate reference-model analysis
    (notebooks/DUP__distribution_analysis.ipynb ->
    notebooks/DUP__historical_reference_model.ipynb).

    Checks analyte_code, rpd, mean_conc, and precision_status (used to
    filter/group the reference population) plus the duplicate-specific
    limit columns stat_detection_limit_dup/limiting_repeatability_dup (used
    to compute the allowable-RPD curve), for missing columns, nulls, dtype
    mismatches, and negative limit values. rpd/mean_conc are derived by
    data_loader._derive_precision_metrics(), not sourced from any raw column.
    """
    return _validate_duplicate_like(df, _DUP_REQUIRED_COLUMNS, _DUP_NONNEGATIVE_COLUMNS, "Duplicate")


def validate_replicate_data(df: pd.DataFrame) -> dict:
    """
    Validate replicate-pair data (data_loader.load_qc_data()'s internal
    schema) ahead of the replicate reference-model analysis
    (notebooks/REP__distribution_analysis.ipynb ->
    notebooks/REP__historical_reference_model.ipynb).

    Same checks as validate_duplicate_data, against the non-suffixed limit
    columns stat_detection_limit/limiting_repeatability.
    """
    return _validate_duplicate_like(df, _REP_REQUIRED_COLUMNS, _REP_NONNEGATIVE_COLUMNS, "Replicate")


def validate_srm_data(df: pd.DataFrame) -> dict:
    """
    Validate SRM (reference-material) data ahead of the SRMS candidate
    pipeline (src/detectors/srms_detector.py).

    1. Filters df to the rows routed to SRMS by select_srm_rows(): every
       Standard row that is not an LCS row (std_lot_code != "Sample").
    2. Checks each column required by srms_detector.py's require_columns()
       (as internal names) for missing values and expected dtype
       (parent_value is excluded -- it's loaded but never used afterwards).
    3. Flags an impossible acceptance-limit span (limit_min >= limit_max),
       since UpperLimit - LowerLimit is used as a divisor throughout the
       detector's distance/warning-fallback calculations.
    """
    print(f"[DataValidator] -- SRM (Reference Material) data validation " + "-" * 40)

    srm_df = select_srm_rows(df)
    print(f"  SRM rows (analytical_type=='Standard' & std_lot_code!='Sample'): {len(srm_df):,}")

    missing_columns, null_counts, dtype_issues = _check_required_columns(srm_df, _SRM_REQUIRED_COLUMNS)

    invalid_limits = {}
    if "limit_min" in srm_df.columns and "limit_max" in srm_df.columns:
        lower = pd.to_numeric(srm_df["limit_min"], errors="coerce")
        upper = pd.to_numeric(srm_df["limit_max"], errors="coerce")
        n_bad_span = int((upper <= lower).sum())
        if n_bad_span:
            invalid_limits["limit_min/limit_max"] = n_bad_span
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

def validate_matrix_spike_data(df: pd.DataFrame) -> dict:
    """
    Validate Matrix Spike data ahead of a future matrix-spike detector
    (src/detectors/matrix_spike_detector.py, currently an empty stub).

    1. Filters df to the Matrix Spike subset: analytical_type == "Spike"
       (confirmed via the real data/raw/QC_Anomaly_Training_Data_v2.xlsx
       "SPK(MS) Assessment" sheet -- every row there is already "Spike"/
       QC_TYPE "MS", so this filter is a safe no-op there, but keeps this
       validator correct if ever run against a mixed dataset, matching
       validate_lcs_data's/validate_srm_data's own defensive filtering).
    2. For each column required, checks the subset has no missing values
       and the column's dtype matches what's expected.
    3. Flags an impossible acceptance-limit span (limit_min >= limit_max),
       mirroring validate_srm_data -- the real MS data shares SRM's schema.

    parent_value/recovery is deliberately not required -- see
    _MS_REQUIRED_COLUMNS' comment for why.
    """
    print(f"[DataValidator] -- Matrix Spike data validation " + "-" * 40)

    if "analytical_type" in df.columns:
        ms_df = df[df["analytical_type"] == "Spike"]
    else:
        ms_df = df.iloc[0:0]

    print(f"  Matrix Spike rows (analytical_type=='Spike'): {len(ms_df):,}")

    missing_columns, null_counts, dtype_issues = _check_required_columns(ms_df, _MS_REQUIRED_COLUMNS)

    invalid_limits = {}
    if "limit_min" in ms_df.columns and "limit_max" in ms_df.columns:
        lower = pd.to_numeric(ms_df["limit_min"], errors="coerce")
        upper = pd.to_numeric(ms_df["limit_max"], errors="coerce")
        n_bad_span = int((upper <= lower).sum())
        if n_bad_span:
            invalid_limits["limit_min/limit_max"] = n_bad_span
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
        "n_rows": len(ms_df),
        "missing_columns": missing_columns,
        "null_counts": null_counts,
        "dtype_issues": dtype_issues,
        "invalid_limits": invalid_limits,
    }

