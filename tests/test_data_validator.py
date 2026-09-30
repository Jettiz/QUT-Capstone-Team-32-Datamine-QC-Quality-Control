"""
Tests for src/data_validator.py's lowercase-internal-schema validators:
validate_blank_data, validate_duplicate_data, validate_replicate_data,
validate_matrix_spike_data. These four check data_loader.load_qc_data()'s
output shape (lowercase internal columns) -- see the module docstring's
"Column-naming convention" section for why these four differ from
validate_lcs_data/validate_srm_data (which stay on raw uppercase columns,
matching LCSDetector/SRMSDetector, and are covered by test_control_detector.py
/ test_srms_detector.py instead).
"""

import pandas as pd
import pytest

from src.data_validator import (
    validate_blank_data,
    validate_duplicate_data,
    validate_replicate_data,
    validate_matrix_spike_data,
)


# ── Blank ────────────────────────────────────────────────────────────────

def _blank_row(**overrides):
    row = {
        "analytical_type": "Blank",
        "std_lot_code": "LOT1",
        "std_code": "STD1",
        "scheme_code": "GE_ICP40Q12",
        "job_code": "J1",
        "analyte_code": "Pb",
        "analysed_date": pd.Timestamp("2024-01-01"),
        "measured_value": 0.02,
        "target_value": 0.0,
        "limit_min": -0.5,
        "limit_max": 0.5,
        "unit_code": "MG_KG",
    }
    row.update(overrides)
    return row


def test_validate_blank_data_passes_on_well_formed_rows():
    df = pd.DataFrame([_blank_row(), _blank_row(analyte_code="Cd")])
    result = validate_blank_data(df)
    assert result["status"] is True
    assert result["n_rows"] == 2
    assert result["missing_columns"] == []
    assert result["null_counts"] == {}


def test_validate_blank_data_filters_to_blank_rows_case_insensitively():
    df = pd.DataFrame([_blank_row(analytical_type="blank"), _blank_row(analytical_type="Standard")])
    result = validate_blank_data(df)
    assert result["n_rows"] == 1


def test_validate_blank_data_flags_missing_column():
    df = pd.DataFrame([_blank_row()]).drop(columns=["limit_max"])
    result = validate_blank_data(df)
    assert result["status"] is False
    assert "limit_max" in result["missing_columns"]


def test_validate_blank_data_flags_null_in_required_column():
    df = pd.DataFrame([_blank_row(), _blank_row(analyte_code=None)])
    result = validate_blank_data(df)
    assert result["status"] is False
    assert result["null_counts"]["analyte_code"] == 1


def test_validate_blank_data_does_not_require_instrument_or_warning_limits():
    # Confirmed against real data/raw/ResultSet.csv: instrument_id and the
    # warning-limit columns are essentially never populated for Blank rows,
    # so they must not be part of the required set at all.
    df = pd.DataFrame([_blank_row()])
    assert "instrument_id" not in df.columns
    assert "limit_max_warning" not in df.columns
    result = validate_blank_data(df)
    assert result["status"] is True


def test_validate_blank_data_no_analytical_type_column_returns_empty_slice():
    df = pd.DataFrame([{"measured_value": 1.0}])
    result = validate_blank_data(df)
    assert result["n_rows"] == 0
    assert result["status"] is False


# ── Duplicate / Replicate ───────────────────────────────────────────────

def _dup_row(**overrides):
    row = {
        "analyte_code": "Cu",
        "rpd": 5.2,
        "mean_conc": 100.0,
        "precision_status": "Pass",
        "stat_detection_limit_dup": 0.5,
        "limiting_repeatability_dup": 10.0,
    }
    row.update(overrides)
    return row


def _rep_row(**overrides):
    row = {
        "analyte_code": "Cu",
        "rpd": 5.2,
        "mean_conc": 100.0,
        "precision_status": "Pass",
        "stat_detection_limit": 0.5,
        "limiting_repeatability": 10.0,
    }
    row.update(overrides)
    return row


def test_validate_duplicate_data_passes_on_well_formed_rows():
    df = pd.DataFrame([_dup_row(), _dup_row(analyte_code="Zn")])
    result = validate_duplicate_data(df)
    assert result["status"] is True
    assert result["n_rows"] == 2


def test_validate_replicate_data_passes_on_well_formed_rows():
    df = pd.DataFrame([_rep_row(), _rep_row(analyte_code="Zn")])
    result = validate_replicate_data(df)
    assert result["status"] is True
    assert result["n_rows"] == 2


def test_validate_duplicate_data_flags_negative_limit():
    df = pd.DataFrame([_dup_row(stat_detection_limit_dup=-1.0)])
    result = validate_duplicate_data(df)
    assert result["status"] is False
    assert result["invalid_limits"]["stat_detection_limit_dup"] == 1


def test_validate_replicate_data_flags_negative_limit():
    df = pd.DataFrame([_rep_row(limiting_repeatability=-2.0)])
    result = validate_replicate_data(df)
    assert result["status"] is False
    assert result["invalid_limits"]["limiting_repeatability"] == 1


def test_validate_duplicate_data_requires_dup_suffixed_columns_not_rep():
    # A row shaped for Replicate (non-_dup limit columns) should be reported
    # as missing the Duplicate-specific (_dup) ones -- confirms the two
    # validators are not accidentally interchangeable.
    df = pd.DataFrame([_rep_row()])
    result = validate_duplicate_data(df)
    assert "stat_detection_limit_dup" in result["missing_columns"]
    assert "limiting_repeatability_dup" in result["missing_columns"]


def test_validate_duplicate_data_flags_missing_rpd():
    # rpd/mean_conc are derived (data_loader._derive_precision_metrics), not
    # sourced -- a frame that never went through that derivation should be
    # reported as missing them, not silently treated as valid.
    df = pd.DataFrame([_dup_row()]).drop(columns=["rpd", "mean_conc"])
    result = validate_duplicate_data(df)
    assert result["status"] is False
    assert "rpd" in result["missing_columns"]
    assert "mean_conc" in result["missing_columns"]


# ── Matrix Spike ─────────────────────────────────────────────────────────

def _ms_row(**overrides):
    row = {
        "analytical_type": "Spike",
        "std_lot_code": "LOT1",
        "std_code": "STD1",
        "job_code": "J1",
        "scheme_code": "GE_ICP40Q12",
        "analyte_code": "Cu",
        "analysed_date": pd.Timestamp("2024-01-01"),
        "measured_value": 105.0,
        "target_value": 100.0,
        "limit_min": 75.0,
        "limit_max": 125.0,
        "limit_min_inclusive": "Y",
        "limit_max_inclusive": "Y",
        "limit_max_warning": 115.0,
        "limit_min_warning": 85.0,
        "limit_min_warning_inclusive": "Y",
        "limit_max_warning_inclusive": "Y",
        "unit_code": "MG_KG",
        "specification_code": "SPEC1",
    }
    row.update(overrides)
    return row


def test_validate_matrix_spike_data_passes_on_well_formed_rows():
    df = pd.DataFrame([_ms_row(), _ms_row(analyte_code="Zn")])
    result = validate_matrix_spike_data(df)
    assert result["status"] is True
    assert result["n_rows"] == 2


def test_validate_matrix_spike_data_filters_to_spike_rows():
    df = pd.DataFrame([_ms_row(), _ms_row(analytical_type="Standard")])
    result = validate_matrix_spike_data(df)
    assert result["n_rows"] == 1


def test_validate_matrix_spike_data_flags_inverted_limit_span():
    df = pd.DataFrame([_ms_row(limit_min=150.0, limit_max=100.0)])
    result = validate_matrix_spike_data(df)
    assert result["status"] is False
    assert result["invalid_limits"]["limit_min/limit_max"] == 1


def test_validate_matrix_spike_data_does_not_require_parent_value_or_instrument():
    # Confirmed against the real data/raw/QC_Anomaly_Training_Data_v2.xlsx
    # "SPK(MS) Assessment" sheet: parent_value (recovery) and instrument_id
    # are both 0% populated there, matching MS_Detection.ipynb's own
    # documented finding that recovery "cannot be computed" from this data.
    df = pd.DataFrame([_ms_row()])
    assert "parent_value" not in df.columns
    assert "instrument_id" not in df.columns
    result = validate_matrix_spike_data(df)
    assert result["status"] is True


def test_validate_matrix_spike_data_no_analytical_type_column_returns_empty_slice():
    df = pd.DataFrame([{"measured_value": 1.0}])
    result = validate_matrix_spike_data(df)
    assert result["n_rows"] == 0
    assert result["status"] is False
